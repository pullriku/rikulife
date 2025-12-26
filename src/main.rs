use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    },
};
use rand::Rng;
use ratatui::{
    prelude::*,
    widgets::{
        Block, Borders, Paragraph,
        canvas::{Canvas, Rectangle},
    },
};
use std::{io, time::Duration};

// 自分で作ったモジュールたち
mod agent;
mod brain;
mod world;

// ※定数は world.rs か consts.rs にある想定
// ここでは簡易的に直書きしてるけど、適宜 use してね
use crate::world::{Position, World};

fn main() -> io::Result<()> {
    // 1. ターミナルのセットアップ (Ratatuiのおまじない)
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 2. 世界の創造 🌍
    // シード値は何でもいいけど、固定すると再現性が取れるよ
    let mut world = World::new(42);

    // 初期エージェントを50匹くらい撒く
    let mut rem: usize = 100;
    while rem > 0 {
        let x = world.rng.random_range(0..crate::world::WIDTH);
        let y = world.rng.random_range(0..crate::world::HEIGHT);
        if world.add_new_agent(Position { x, y }).is_some() {
            rem -= 1;
        }

        if rem == 0 {
            break;
        }
    }

    for _ in 0..5000 {
        world.spawn_foods();
    }

    run_app(&mut terminal, &mut world.clone()).unwrap();

    // 4. お片付け (終了処理)
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    println!();

    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, world: &mut World) -> io::Result<()> {
    #[allow(unused_mut)]
    let mut last_tick = std::time::Instant::now();
    let tick_rate = Duration::from_millis(50); // 更新速度 (50ms = 20fps)

    loop {
        // --- 描画フェーズ 🎨 ---
        terminal.draw(|f| ui(f, world))?;

        // --- 入力 & 更新フェーズ 🎮 ---
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        // キー入力があれば処理、なければ待機
        if crossterm::event::poll(timeout)?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') => return Ok(()), // 'q' で終了
                KeyCode::Char(' ') => {
                    // スペースキーでポーズとか入れたいならここに
                }
                _ => {}
            }
        }

        // 時間が経ったら World を1ステップ進める
        // if last_tick.elapsed() >= tick_rate {
        //     world.step();
        //     last_tick = std::time::Instant::now();
        // }

        world.step();
    }
}

// --- UI構築ロジック 🖼️ ---
fn ui(f: &mut Frame, world: &World) {
    // 画面を左右に分割
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70), // 左70%: マップ
            Constraint::Percentage(30), // 右30%: 情報
        ])
        .split(f.area());

    // --- 1. 左側: 世界の描画 (Canvas) ---
    // Canvasウィジェットを使うと、座標指定で矩形を描けるので便利！
    let canvas = Canvas::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Artificial Life "),
        )
        .x_bounds([0.0, crate::world::WIDTH as f64])
        .y_bounds([0.0, crate::world::HEIGHT as f64])
        .paint(|ctx| {
            // A. 餌を描画 (緑色の小さな点) 🍏
            for y in 0..crate::world::HEIGHT {
                for x in 0..crate::world::WIDTH {
                    if world.foods[y][x] {
                        let (draw_x, draw_y) = calc_draw_position(Position { x, y });
                        ctx.draw(&Rectangle {
                            x: draw_x,
                            y: draw_y,
                            width: 1.0,
                            height: 1.0,
                            color: Color::Green,
                        });
                    }
                }
            }

            // B. エージェントを描画 (RGB色の四角形)
            for agent in world.agents.values() {
                // Agentの色 (0.0~1.0) を u8 (0~255) に変換
                let r = (agent.color[0] * 255.0) as u8;
                let g = (agent.color[1] * 255.0) as u8;
                let b = (agent.color[2] * 255.0) as u8;

                let (draw_x, draw_y) = calc_draw_position(agent.pos);

                ctx.draw(&Rectangle {
                    x: agent.pos.x as f64,
                    y: (crate::world::HEIGHT - 1 - agent.pos.y) as f64,
                    width: 1.0,
                    height: 1.0,
                    color: Color::Rgb(r, g, b),
                });

                if let Some(action) = agent.last_action {
                    match action {
                        crate::agent::Action::Attack => {
                            // 攻撃してる時は赤い "x" を重ねる
                            ctx.print(
                                draw_x,
                                draw_y,
                                Span::styled(
                                    "x",
                                    Style::default()
                                        .fg(Color::LightBlue)
                                        .add_modifier(Modifier::BOLD),
                                ),
                            );
                        }
                        crate::agent::Action::Heal => {
                            // 回復してる時は緑の "+" を重ねる
                            ctx.print(
                                draw_x,
                                draw_y,
                                Span::styled(
                                    "+",
                                    Style::default()
                                        .fg(Color::LightGreen)
                                        .add_modifier(Modifier::BOLD),
                                ),
                            );
                        }
                        _ => {
                            // 移動や待機の時は、記号を出さずにRGBの色だけ見せる
                            // (何も描画しない)
                        }
                    }
                }
            }
        });

    f.render_widget(canvas, chunks[0]);

    // --- 2. 右側: 統計情報 (Paragraph) ---
    let population = world.agents.len();
    let max_gen = world
        .agents
        .values()
        .map(|a| a.generation)
        .max()
        .unwrap_or(0);
    let total_energy: u32 = world.agents.values().map(|a| a.energy).sum();
    let avg_energy = if population > 0 {
        total_energy / population as u32
    } else {
        0
    };
    let food_count: usize = world
        .foods
        .iter()
        .map(|row| row.iter().filter(|&&f| f).count())
        .sum();

    let info_text = vec![
        Line::from(vec![Span::raw("Statistics 📊")]),
        Line::from(""),
        Line::from(vec![Span::raw(format!("Step: {}", world.step))]),
        Line::from(vec![Span::styled(
            format!("Population: {}", population),
            Style::default().fg(Color::Yellow),
        )]),
        Line::from(vec![Span::raw(format!("Max Generation: {}", max_gen))]),
        Line::from(vec![Span::raw(format!("Avg Energy: {}", avg_energy))]),
        Line::from(""),
        Line::from(vec![Span::styled(
            format!("Food Count: {}", food_count),
            Style::default().fg(Color::Green),
        )]),
        Line::from(""),
        Line::from("Controls:"),
        Line::from(" 'q' to Quit"),
    ];

    let info_block = Paragraph::new(info_text)
        .block(Block::default().borders(Borders::ALL).title(" Info "));

    f.render_widget(info_block, chunks[1]);
}

fn calc_draw_position(pos: crate::world::Position) -> (f64, f64) {
    let draw_x = pos.x as f64;
    let draw_y = (crate::world::HEIGHT - 1 - pos.y) as f64;
    (draw_x, draw_y)
}
