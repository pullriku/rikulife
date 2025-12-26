use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::{
    brain::{Brain, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE},
    world::{AgentId, CHILD_INIT_ENERGY, INIT_ENERGY, LIFESPAN_RANGE, MAX_ENERGY, Position},
};

pub type Color = [f32; 3];

#[derive(Debug, Clone)]
pub struct Agent {
    pub(crate) id: AgentId,
    pub(crate) pos: Position,
    pub(crate) energy: u32,
    pub(crate) max_energy: u32,
    pub generation: u32,

    pub(crate) brain: Brain,

    pub(crate) color: Color,

    pub(crate) last_action: Option<Action>,

    pub(crate) age: u32,
    /// 寿命（この歳になったら死ぬ）
    pub(crate) lifespan: u32,
}

impl Agent {
    /// ランダムな個体を生成。
    /// 最初のアダムとイブ用。
    pub fn new_random<R: Rng + ?Sized>(id: usize, pos: Position, rng: &mut R) -> Self {
        // 重みを正規分布で初期化
        let w1 = random_matrix(HIDDEN_SIZE, INPUT_SIZE, rng);
        let b1 = Array1::zeros(HIDDEN_SIZE);
        let w2 = random_matrix(OUTPUT_SIZE, HIDDEN_SIZE, rng);
        let b2 = Array1::zeros(OUTPUT_SIZE);

        let brain = Brain::new(w1, b1, w2, b2);

        Self {
            id,
            pos,
            energy: INIT_ENERGY,
            max_energy: MAX_ENERGY,
            generation: 1,
            brain,
            color: [rng.random(), rng.random(), rng.random()],
            last_action: None,
            age: 0,
            lifespan: rng.random_range(LIFESPAN_RANGE),
        }
    }

    /// 子供を生成する
    /// - new_id: 新しいID
    /// - new_pos: 生まれる場所
    /// - rng: 乱数生成器
    pub fn new_child<R: Rng + ?Sized>(
        &self,
        new_id: usize,
        new_pos: Position,
        rng: &mut R,
    ) -> Self {
        // 1. 脳の遺伝と変異
        // Brain::spawn_child を呼び出す。
        // rate: 1.0 (全パラメータを変異させる「ドリフト」方式を採用)
        // sigma: 0.02 (親の値を少しだけズラす)
        let child_brain = self.brain.spawn_child(1.0, 0.2, rng);

        // 2. 最大エネルギー(体格)の遺伝と変異
        // 親の値を基準に ±5 の範囲でランダムに変化させる
        // 極端になりすぎないように .clamp(50, 200) で制限をかける
        let mutation_range = 5;
        let diff = rng.random_range(-mutation_range..=mutation_range);
        let child_max_energy = (self.max_energy as i32 + diff).clamp(10, 500) as u32;

        Self {
            id: new_id,
            pos: new_pos,

            // 生まれたての状態設定
            energy: CHILD_INIT_ENERGY, // 子供の初期体力（親のコスト50と同じにして等価交換にする）
            max_energy: child_max_energy,
            generation: self.generation + 1, // 世代を1つ進める

            brain: child_brain,

            // 色はとりあえず親と同じ色で初期化
            // (動き始めれば Brain の出力によってすぐに自分の色に変わるよ！)
            color: self.color,
            last_action: None,

            age: 0,
            lifespan: rng.random_range(LIFESPAN_RANGE),
        }
    }
}

/// ランダム行列を作る
fn random_matrix<R: Rng + ?Sized>(rows: usize, cols: usize, rng: &mut R) -> Array2<f32> {
    let dist = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| dist.sample(rng))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
    Stay = 4,
    Attack = 5,
    Heal = 6,
}

impl Action {
    // 確率(出力)の配列から、一番値が大きい行動を選ぶ
    pub fn from_output(output: &[f32]) -> Self {
        // 0~6番目の要素の中で最大値のインデックスを探す
        let (index, _) = output
            .iter()
            .take(7) // 最初の7つが行動
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((4, &0.0)); // エラーならStay

        match index {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            4 => Action::Stay,
            5 => Action::Attack,
            6 => Action::Heal,
            _ => Action::Stay,
        }
    }
}
