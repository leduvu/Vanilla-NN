//#############################################################################
// Author:      Le Duyen Sandra Vu
// Matr.Nr.:    4171456
// Compiler:    rustc 1.27.0 (3eda71b00 2018-06-19)
// Cargo:       1.27.0 (1e95190e5 2018-05-27)
// OS:          MacOS 10.13.4
// Subject:     Single hidden layer neural network for classification
//
// NN.VERSION 29.08.18
//#############################################################################

/*!
 * has functions to train and evaluate a single hidden layer neural network.
 */

use ndarray::arr1;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Range;

use std::f32;

/// # Remarks
/// The number of output neurons is set to 1 (nn was not testet for other numbers).
const O_NEURONS: usize = 1;

/// # Remarks
/// * `w_ih`: Matrix weights from input to hidden
/// * `w_ho`: Matrix weights from hidden to output
/// * `b_h`: Vector bias hidden
/// * `b_o`: Vector bias output
/// * `input_len`: Number of features aka input neurons
/// * `h_neurons`: Number of hidden neurons
/// * `learning_r`: Learning rate
pub struct Settings {
    w_ih: Array2<f32>, // Weights from input to hidden
    w_ho: Array2<f32>, // Weights from hidden to output
    b_h: Array1<f32>,  // Bias hidden
    b_o: Array1<f32>,  // Bias output
    input_len: usize,  // Number of features aka input neurons

    h_neurons: usize, // Number of hidden neurons
    learning_r: f32,  // Learning rate
}

impl Settings {
    // Initialize weights with random numbers between 0 and 1
    pub fn new(feature_neurons: usize, h_neurons: usize, learning_r: f32) -> Self {
        Settings {
            w_ih: Array2::random((feature_neurons, h_neurons), Range::new(0., 1.)),
            w_ho: Array2::random((h_neurons, O_NEURONS), Range::new(0., 1.)),

            b_h: Array1::random(h_neurons, Range::new(0., 1.)),
            b_o: Array1::random(O_NEURONS, Range::new(0., 1.)),
            input_len: feature_neurons,
            h_neurons,
            learning_r,
        }
    }
}

/// # Remarks
/// Sigmoid function for Array1.
pub fn sig(mut arr: Array1<f32>) -> Array1<f32> {
    for value in arr.iter_mut() {
        *value = 1.0 / (1.0 + (-*value).exp());
    }
    arr
}

/// # Remarks
/// Derivativ of the sigmoid function for Array1.
///
/// # Warning
/// The input array has to have the sigmoid values.
/// ```sig_prime(sig(array![0.0, -1.0, 2.0, -0.3]));```
pub fn sig_prime(mut arr: Array1<f32>) -> Array1<f32> {
    for value in arr.iter_mut() {
        *value = *value * (1.0 - *value);
    }
    arr
}

/// # Remarks
/// Tanh function for Array1.
pub fn tanh(mut arr: Array1<f32>) -> Array1<f32> {
    for value in arr.iter_mut() {
        *value = value.tanh();
    }
    arr
}

/// # Remarks
/// Derivativ of the tanh function for Array1.
///
/// # Warning
/// The input array has to have the sigmoid values.
/// ```tanh_prime(tanh(array![0.0, -1.0, 2.0, -0.3]));```
pub fn tanh_prime(mut arr: Array1<f32>) -> Array1<f32> {
    for value in arr.iter_mut() {
        *value = 1. - value.powi(2);
    }
    arr
}

/// # Arguments
/// * `train_data`: [line][column1 value, column2 value, ...]
/// * `nn`: Struct which contains weights, biases, ...
/// * `f`: Activation function
/// * `f_prime`: Derivative of the activation function
///
/// # Remarks
/// Updates the weights and biases, which where initialized before training and
/// passed as part of the second argument (Settings contains weights, biases, ...).
pub fn train<'a>(
    train_data: &Vec<Vec<f32>>,
    nn: &'a mut Settings,
    f: &Fn(Array1<f32>) -> Array1<f32>,
    f_prime: &Fn(Array1<f32>) -> Array1<f32>,
) {
    let mut delta_o: Array1<f32> = Array1::zeros(train_data.len());
    let mut input_f: Array2<f32> = Array2::zeros((train_data.len(), nn.input_len));
    let mut slope_h: Array2<f32> = Array2::zeros((train_data.len(), nn.h_neurons));
    let mut h_result: Array2<f32> = Array2::zeros((train_data.len(), nn.h_neurons));

    for (ii, line) in train_data.iter().enumerate() {
        let features: Array1<f32> = arr1(&line[1..]);
        let label: Array1<f32> = arr1(&line[0..1]);

        // Hidden layer output (with activation function),
        // Output layer output (with activation function),
        // Gradient of Error, slope at output layer and delta output
        let h_calc: Array1<f32> = f(features.dot(&nn.w_ih) + &nn.b_h); // f(inputVec * weightMatrix + biasVec)
        let o_calc: Array1<f32> = f(h_calc.dot(&nn.w_ho) + &nn.b_o); // f(hiddenVec * weightMatrix + biasVec)
        let err: Array1<f32> = &label - &o_calc;
        let slope_o: Array1<f32> = f_prime(o_calc);
        delta_o[ii] = err[0] * slope_o[0];

        // Save the features/input (as matrix)
        // [ [i11, i12, i13,, ...]
        //   [i21, i22, i23, ...]
        //    ..]
        for (jj, v) in features.iter().enumerate() {
            input_f[[ii, jj]] = *v;
        }

        // Save hidden result array (as matrix)
        // [ [i1h1, i1h2, i1,h3, ...]
        //   [i2h1, i2h2, i2h3, ...]
        //    ..]
        for (jj, v) in h_calc.iter().enumerate() {
            h_result[[ii, jj]] = *v;
        }

        // Save the slopes at hidden layer (as matrix)
        let mut tmp: Array1<f32> = f_prime(h_calc);
        for (jj, v) in tmp.iter().enumerate() {
            slope_h[[ii, jj]] = *v;
        }
    }

    let d_o: Array1<f32> = delta_o.clone();
    let nn_ho_t: Array2<f32> = nn.w_ho.clone().reversed_axes(); // Transpose w_ho

    // Error at hidden layer and delta hidden
    let err_h: Array2<f32> = delta_o
        .into_shape((train_data.len(), 1))
        .unwrap()
        .dot(&nn_ho_t);
    let d_h: Array2<f32> = err_h * slope_h;

    // Weights and bias update
    nn.b_h = &nn.b_h + &d_h.scalar_sum() * nn.learning_r; // biasVec + float * float
    nn.b_o = &nn.b_o + &d_o.scalar_sum() * nn.learning_r;
    nn.w_ih = &nn.w_ih + &(input_f.reversed_axes().dot(&d_h) * nn.learning_r); // weightMatrix + Matrix
    nn.w_ho = &nn.w_ho
        + &(h_result
            .reversed_axes()
            .dot(&d_o.into_shape((train_data.len(), 1)).unwrap()) * nn.learning_r);
}

/// # Arguments
/// * `train_data`: [line][column1 value, column2 value, ...]
/// * `nn`: Struct which contains weights, biases, ...
/// * `f`: Activation function
///
/// # Remarks
/// Uses the data, weights and biases to calculate the output/label.
/// Then compares the calculated label with the expected label and calculates the accuracy.
/// Note: calculated values under 0.1 get the label 0.0 and calculated values over 0.9 get the label 1.0
pub fn eval(test_data: &Vec<Vec<f32>>, nn: Settings, f: &Fn(Array1<f32>) -> Array1<f32>) {
    let mut count: f32 = 0.;

    // Go through each line
    // [label, feature1, feature2, ...] ==
    // [Output, InputNeuron1, InputNeuron2, ...]
    // Currently the label/output is always a single float number
    for line in test_data.iter() {
        let features: Array1<f32> = arr1(&line[O_NEURONS..]); // First value is the label
        let label: Array1<f32> = arr1(&line[0..O_NEURONS]); // Rest are the features

        // Hidden layer output and output layer output (with activation function)
        let h_calc: Array1<f32> = f(features.dot(&nn.w_ih) + &nn.b_h); // f(inputVec * weightMatrix + biasVec)
        let o_calc: Array1<f32> = f(h_calc.dot(&nn.w_ho) + &nn.b_o); // f(hiddenVec * weightMatrix + biasVec)

        let mut calc = -10.0; // Random initialization for calculated label

        // Calculated values under 0.1 will be 0.0
        // Calculated values over 0.9 will be 1.0
        // All other values stay the same.
        if o_calc[0] < 0.1 {
            calc = 0.0;
        }
        if o_calc[0] > 0.9 {
            calc = 1.0
        }

        // Count right guesses
        if calc == label[0] {
            count += 1.;
        }
        println!(" {0: <15}--- {1: <15}", o_calc[0], label[0]);
    }
    println!("Accuracy: {:?}%", (count / test_data.len() as f32) * 100.0);
}

#[cfg(test)]
mod tests {
    use super::{sig, sig_prime, tanh, tanh_prime};
    use float_cmp::*;
    #[test]
    fn sig_test() {
        let a = array![0.5, 0.26894142, 0.88079707, 0.42555748];
        let b = sig(array![0.0, -1.0, 2.0, -0.3]);

        for ii in 0..a.len() {
            println!("{:?}", b[ii]);

            assert!(a[ii].approx_eq_ulps(&b[ii], 8));
        }
    }

    #[test]
    fn sig_prime_test() {
        let a = array![0.25, 0.19661193, 0.10499358, 0.24445831];
        let b = sig_prime(sig(array![0.0, -1.0, 2.0, -0.3]));

        for ii in 0..a.len() {
            println!("{:?}", b[ii]);
            assert!(a[ii].approx_eq_ulps(&b[ii], 8));
        }
    }

    #[test]
    fn tanh_test() {
        assert_eq!(
            array![0f32.tanh(), -1f32.tanh(), 2f32.tanh(), -0.3f32.tanh()],
            tanh(array![0.0, -1.0, 2.0, -0.3])
        );
    }

    #[test]
    fn tanh_prime_test() {
        let a = array![1.0, 0.41997434, 0.07065082, 0.91513696];
        let b = tanh_prime(tanh(array![0.0, -1.0, 2.0, -0.3]));

        for ii in 0..a.len() {
            println!("{:?}", b[ii]);
            assert!(a[ii].approx_eq_ulps(&b[ii], 8));
        }
    }
}
