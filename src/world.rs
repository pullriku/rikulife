use std::{collections::HashMap, ops::Range};

use ndarray::Array1;
use rand::{Rng, SeedableRng, seq::IndexedRandom};

use crate::{
    agent::{Action, Agent, Color},
    brain::{INPUT_FIELD_LENGTH, INPUT_SIZE},
};

pub type AgentId = usize;

pub const WIDTH: usize = 50;
pub const HEIGHT: usize = 50;
pub const MAX_FOODS: usize = 2500;
pub const MAX_ENERGY: u32 = 100;
pub const INIT_ENERGY: u32 = MAX_ENERGY / 10 * 5;

pub const CHILD_INIT_ENERGY: u32 = MAX_ENERGY / 10 * 5;
pub const REPRODUCE_COST: u32 = MAX_ENERGY / 10 * 7;

/// 餌を1ステップに何回湧かせようとするか
pub const FOOD_SPAWN_COUNT_SUMMER: usize = 250;
pub const FOOD_SPAWN_COUNT_WINTER: usize = 100;
pub const FOOD_ENERGY: u32 = 60;

/// 攻撃、回復にかかるコスト
pub const INTERACT_COST: u32 = 10;
/// 攻撃の相手の体力の変化量（吸血の場合は、これに手数料を引いたものをゲットできる）
pub const ATTACK_AMOUNT: i32 = -20;
/// 回復の相手の体力の変化量
pub const HEAL_AMOUNT: u32 = 8;

pub const LIFESPAN_RANGE: Range<u32> = 500..700;

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Clone)]
pub struct World {
    pub step: u64,
    pub agents: HashMap<AgentId, Agent>,

    pub grid: Vec<Vec<Option<AgentId>>>,
    pub foods: Vec<Vec<bool>>,

    pub rng: rand::rngs::StdRng,
    next_id: usize,
}

impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            step: 0,
            agents: HashMap::new(),
            grid: vec![vec![None; WIDTH]; HEIGHT],
            foods: vec![vec![false; WIDTH]; HEIGHT],
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            next_id: 0,
        }
    }

    pub fn step(&mut self) {
        self.step += 1;

        let dead_ids: Vec<usize> = self
            .agents
            .values()
            .filter(|a| a.energy == 0)
            .map(|a| a.id)
            .collect();

        for id in dead_ids {
            self.remove_agent(id);
        }

        self.spawn_foods();

        let mut agent_ids: Vec<usize> = self.agents.keys().cloned().collect();
        agent_ids.sort_by_key(|id| self.agents[id].energy);

        for id in agent_ids {
            debug_assert!(self.agents.contains_key(&id));

            let (action, new_color) = {
                let input = self.get_input(id);
                let agent = self.agents.get(&id).unwrap();
                let output = agent.brain.forward(&input);

                // 出力から行動と色を決定
                let act = Action::from_output(output.as_slice().unwrap());
                let r = output[7].clamp(0.0, 1.0);
                let g = output[8].clamp(0.0, 1.0);
                let b = output[9].clamp(0.0, 1.0);
                (act, [r, g, b])
            };

            if let Some(agent) = self.agents.get_mut(&id) {
                agent.last_action = Some(action);

                agent.age += 1;
                if agent.age >= agent.lifespan {
                    agent.energy = 0;
                }
            }

            self.apply_action(id, action, new_color);

            self.try_reproduce(id);
        }
    }

    /// エージェントを世界に追加するヘルパー
    #[must_use]
    pub fn add_new_agent(&mut self, pos: Position) -> Option<()> {
        if self.grid[pos.y][pos.x].is_some() {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        let agent = Agent::new_random(id, pos, &mut self.rng);

        // 空間と実体の両方に登録
        self.add_agent(agent, pos);

        Some(())
    }

    fn add_agent(&mut self, agent: Agent, pos: Position) {
        self.grid[pos.y][pos.x] = Some(agent.id);
        self.agents.insert(agent.id, agent);
    }

    fn remove_agent(&mut self, id: AgentId) {
        let agent = self.agents.remove(&id).unwrap();
        self.grid[agent.pos.y][agent.pos.x] = None;
    }

    // 餌を生成する処理
    /// - 中央に近いほど湧きやすい
    /// - MAX_FOODSを超えたら湧かない
    pub fn spawn_foods(&mut self) {
        // 1. 現在の餌の総数を数える (Maxチェック用)
        let current_food_count: usize = self
            .foods
            .iter()
            .map(|row| row.iter().filter(|&&has_food| has_food).count())
            .sum();

        // 既に満タンなら何もしない
        if current_food_count >= MAX_FOODS {
            return;
        }

        // --- 設定値 ---
        // let population = self.agents.len();
        // let spawn_count = 50 + (population / 2);
        let base_probability = 0.2; // チャンスが来た時の基本確率 (20%)

        // 中心座標と、中心から角までの最大距離 (正規化用)
        let center_x = WIDTH as f32 / 2.0;
        let center_y = HEIGHT as f32 / 2.0;
        let max_dist = (center_x.powi(2) + center_y.powi(2)).sqrt();

        let is_winter = (self.step / 2000) % 2 == 1;

        let spawn_count = if is_winter {
            FOOD_SPAWN_COUNT_WINTER
        } else {
            FOOD_SPAWN_COUNT_SUMMER
        };

        for _ in 0..spawn_count {
            // ランダムな座標を選ぶ
            let x = self.rng.random_range(0..WIDTH);
            let y = self.rng.random_range(0..HEIGHT);

            // 既に餌がある場所はスキップ
            if self.foods[y][x] {
                continue;
            }

            // 2. 確率計算：中心に近いほど高確率にする
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx.powi(2) + dy.powi(2)).sqrt();

            // 距離スコア (0.0 ~ 1.0)
            // 中心(dist=0)なら 1.0, 一番遠い角(dist=max)なら 0.0
            let score = (1.0 - (dist / max_dist)).max(0.0);

            // 最終確率 = 基本確率 * スコアの2乗
            // (2乗することで、中心付近に急激に集まる分布になる！)
            let probability = base_probability * score.powi(2);

            // 3. 乱数で判定
            if self.rng.random::<f32>() < probability {
                self.foods[y][x] = true;
            }
        }
    }

    /// エージェントIDを受け取り、その視界データ(150次元)を返す
    pub fn get_input(&self, id: AgentId) -> Array1<f32> {
        let agent = self.agents.get(&id).expect("Agent not found");
        let (center_x, center_y): (isize, isize) = (
            agent.pos.x.try_into().unwrap(),
            agent.pos.y.try_into().unwrap(),
        );

        let mut input = Vec::with_capacity(INPUT_SIZE);

        let radius = (INPUT_FIELD_LENGTH / 2) as isize;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let nx = center_x + dx;
                let ny = center_y + dy;

                // 1. 壁判定 (範囲外なら壁)
                let is_wall =
                    nx < 0 || ny < 0 || nx >= WIDTH as isize || ny >= HEIGHT as isize;

                // 範囲内の情報を取得
                let mut is_food = false;
                let mut is_agent = false;
                let mut color = [0.0; 3];

                if !is_wall {
                    let (ux, uy) = (nx as usize, ny as usize);
                    is_food = self.foods[uy][ux];

                    if let Some(target_id) = self.grid[uy][ux]
                        && target_id != id
                    {
                        is_agent = true;
                        // 相手の色を取得
                        if let Some(target) = self.agents.get(&target_id) {
                            color = target.color;
                        }
                    }
                }

                // 入力ベクトルに追加 (6要素)
                input.push(if is_wall { 1.0 } else { 0.0 });
                input.push(if is_food { 1.0 } else { 0.0 });
                input.push(if is_agent { 1.0 } else { 0.0 });
                input.push(color[0]); // R
                input.push(color[1]); // G
                input.push(color[2]); // B
            }
        }

        // 入力ベクトルの長さを確認
        debug_assert_eq!(input.len(), INPUT_SIZE);

        Array1::from(input)
    }

    /// 行動を適用する
    fn apply_action(&mut self, id: AgentId, action: Action, new_color: Color) {
        let Some(agent) = self.agents.get_mut(&id) else {
            panic!("Agent not found");
        };

        agent.color = new_color;
        // 基礎代謝コスト
        agent.energy = agent.energy.saturating_sub(1);

        match action {
            Action::Up | Action::Down | Action::Left | Action::Right => {
                self.move_agent(id, action);
            }
            Action::Stay => {
                // 待機ボーナス（何もしないなら少し消費が減る等のルールを入れてもいい）
            }
            Action::Attack => {
                self.interact_area(id, ATTACK_AMOUNT); // 周囲にダメージ
            }
            Action::Heal => {
                self.interact_area(id, HEAL_AMOUNT as i32); // 周囲を回復（自分はコスト消費）
            }
        }
    }

    /// 移動ロジック
    fn move_agent(&mut self, id: AgentId, action: Action) {
        // 現在位置と移動先を計算
        let Position { x: cx, y: cy } = self.agents.get(&id).map(|a| a.pos).unwrap();
        let (dx, dy) = match action {
            Action::Up => (0, -1),
            Action::Down => (0, 1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
            _ => (0, 0),
        };

        // 移動コスト消費
        if let Some(agent) = self.agents.get_mut(&id) {
            agent.energy = agent.energy.saturating_sub(1); // 移動は疲れる
        }

        let nx = cx as isize + dx;
        let ny = cy as isize + dy;

        // 壁チェック
        if nx < 0 || ny < 0 || nx >= WIDTH as isize || ny >= HEIGHT as isize {
            return; // 範囲外なので移動キャンセル
        }

        let (nx, ny) = (nx as usize, ny as usize);

        // 衝突チェック (誰もいないか？)
        if self.grid[ny][nx].is_none() {
            // 移動処理：グリッドを更新
            self.grid[cy][cx] = None;
            self.grid[ny][nx] = Some(id);

            // エージェントの座標更新
            if let Some(agent) = self.agents.get_mut(&id) {
                agent.pos = Position { x: nx, y: ny };

                // 餌チェック & 自動食事
                if self.foods[ny][nx] {
                    self.foods[ny][nx] = false; // 餌消滅
                    let gain = FOOD_ENERGY; // 回復量
                    agent.energy = (agent.energy + gain).min(agent.max_energy);
                }
            }
        }
    }

    /// 周囲への干渉（攻撃・回復）
    fn interact_area(&mut self, id: AgentId, effect: i32) {
        let Position { x: cx, y: cy } = self.agents.get(&id).map(|a| a.pos).unwrap();

        if let Some(me) = self.agents.get_mut(&id) {
            me.energy = me.energy.saturating_sub(INTERACT_COST);
        }

        // 周囲8マスに作用
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                } // 自分は除外

                let nx = cx as isize + dx;
                let ny = cy as isize + dy;

                if nx >= 0
                    && ny >= 0
                    && nx < WIDTH as isize
                    && ny < HEIGHT as isize
                    && let Some(target_id) = self.grid[ny as usize][nx as usize]
                    && let Some(target) = self.agents.get_mut(&target_id)
                {
                    if effect < 0 {
                        // 攻撃：相手の体力を減らす
                        let damage = effect.unsigned_abs();
                        let actual_damage = target.energy.min(damage); // 相手が持ってる分しか奪えない

                        target.energy = target.energy.saturating_sub(actual_damage);

                        let absorb = (actual_damage as f32 * 0.8) as u32;

                        // ※奪い取るルールにするなら、ここで自分のenergyを増やす
                        if let Some(me) = self.agents.get_mut(&id) {
                            me.energy = (me.energy + absorb).min(me.max_energy);
                        }
                    } else {
                        // 回復：相手の体力を増やす
                        target.energy =
                            (target.energy + effect as u32).min(target.max_energy);
                    }
                }
            }
        }
    }

    pub fn try_reproduce(&mut self, id: AgentId) {
        let (pos, can_reproduce) = {
            if let Some(agent) = self.agents.get(&id) {
                (agent.pos, agent.energy >= agent.max_energy)
            } else {
                return;
            }
        };

        if !can_reproduce {
            return;
        }

        // 2. 繁殖コストの支払い（書き込み）
        // 子供が産めるかどうかに関わらず、エネルギーは消費する（混雑ペナルティ）
        if let Some(parent) = self.agents.get_mut(&id) {
            parent.energy = parent.energy.saturating_sub(REPRODUCE_COST);
        }

        // 3. 産む場所を探す
        // 周囲8マスの空き地リストを作成
        let mut free_spots = Vec::new();
        let Position { x: cx, y: cy } = pos;
        let (cx, cy) = (cx as isize, cy as isize);

        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                } // 自分自身の場所はスキップ

                let nx = cx + dx;
                let ny = cy + dy;

                // 範囲内かチェック
                if nx >= 0 && ny >= 0 && nx < WIDTH as isize && ny < HEIGHT as isize {
                    let (ux, uy) = (nx as usize, ny as usize);
                    // グリッドが空(None)なら候補に入れる
                    if self.grid[uy][ux].is_none() {
                        free_spots.push(Position { x: ux, y: uy });
                    }
                }
            }
        }

        // 4. 子供の生成
        if let Some(child_pos) = free_spots.choose(&mut self.rng).copied() {
            let child = {
                let parent = self.agents.get(&id).unwrap();
                let new_id = self.next_id;
                self.next_id += 1;

                // 親の脳を引き継いだ子供を作る
                parent.new_child(new_id, child_pos, &mut self.rng)
            };

            // 世界に登録
            self.add_agent(child, child_pos);
        }
    }
}
