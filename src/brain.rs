use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// ニューラルネットワークの形状。
pub const INPUT_SIZE: usize = INPUT_FIELD_SIZE * (INPUT_CELL_TYPE_SIZE + RGB_COLOR_SIZE);

pub const INPUT_FIELD_LENGTH: usize = 7;
pub const INPUT_FIELD_SIZE: usize = INPUT_FIELD_LENGTH * INPUT_FIELD_LENGTH;

/// 周囲の状態。壁、餌、他の生命。
pub const INPUT_CELL_TYPE_SIZE: usize = 3;

pub const HIDDEN_SIZE: usize = 64;

pub const OUTPUT_SIZE: usize = OUTPUT_ACTION_SIZE + RGB_COLOR_SIZE;

/// 行動(上下左右、待機、攻撃・お裾分け）
pub const OUTPUT_ACTION_SIZE: usize = 4 + 1 + 2;

/// RGB色
pub const RGB_COLOR_SIZE: usize = 3;

#[derive(Debug, Clone)]
pub struct Brain {
    weights_l1: Array2<f32>,
    biases_l1: Array1<f32>,

    weights_l2: Array2<f32>,
    biases_l2: Array1<f32>,
}

impl Brain {
    pub fn new(
        weights_l1: Array2<f32>,
        biases_l1: Array1<f32>,
        weights_l2: Array2<f32>,
        biases_l2: Array1<f32>,
    ) -> Self {
        Self {
            weights_l1,
            biases_l1,
            weights_l2,
            biases_l2,
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut hidden = self.weights_l1.dot(input) + &self.biases_l1;
        relu_inplace(&mut hidden);
        self.weights_l2.dot(&hidden) + &self.biases_l2
    }

    /// 単為生殖。
    /// 親をコピーして突然変異させた子を返す・
    pub fn spawn_child<R: Rng + ?Sized>(
        &self,
        rate: f32,
        sigma: f32,
        rng: &mut R,
    ) -> Brain {
        let mut child = self.clone();
        child.mutate_inplace(rate, sigma, rng);
        child
    }

    /// 突然変異。
    /// 各パラメータを確率 rate で N(0, sigma) だけ揺らす。
    /// `rate`は突然変異の割合。`sigma`は標準偏差。
    pub fn mutate_inplace<R: Rng + ?Sized>(
        &mut self,
        rate: f32,
        sigma: f32,
        rng: &mut R,
    ) {
        debug_assert!((0.0..=1.0).contains(&rate));

        let mut mutate_val = |val: &mut f32| {
            if rng.random::<f32>() < rate {
                // StandardNormal は 平均0, 標準偏差1 の乱数を出す
                // それに sigma を掛ければ N(0, sigma) になる
                let noise: f32 = StandardNormal.sample(rng);
                *val += noise * sigma;
            }
        };

        for v in self.weights_l1.iter_mut() {
            mutate_val(v);
        }
        for v in self.biases_l1.iter_mut() {
            mutate_val(v);
        }
        for v in self.weights_l2.iter_mut() {
            mutate_val(v);
        }
        for v in self.biases_l2.iter_mut() {
            mutate_val(v);
        }
    }
}

fn relu_inplace(x: &mut Array1<f32>) {
    x.mapv_inplace(|v| v.max(0.0));
}
