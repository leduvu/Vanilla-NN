//#############################################################################
// Author:      Le Duyen Sandra Vu
// Matr.Nr.:    4171456
// Compiler:    rustc 1.27.0 (3eda71b00 2018-06-19)
// Cargo:       1.27.0 (1e95190e5 2018-05-27)
// OS:          MacOS 10.13.4
// Subject:     Single hidden layer neural network for classification
//
// MAIN.VERSION 29.08.18
//
// EXAMPLE:
// $ target/release/neural_network xor.txt xor.txt -i 2000 -t
// TRAINING
// iterations=2000 hidden_neurons=3 learning_rate=0.1
//
// EVALUATION
//  0.9908207      --- 1
//  0.9891633      --- 1
//  0.000102072954 --- 0
// Accuracy: 100.0%
//#############################################################################

/*!
 * Neral Network Program:
 * Training including a final evaluation (accuracy).
 */

extern crate clap;
extern crate failure;
extern crate float_cmp;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate stdinout;
use ndarray::arr1;
use ndarray::Array1;
use stdinout::OrExit;

pub mod args;
use args::parse_args;

pub mod nn;
pub mod reader;
use nn::*;

/// # Arguments (Terminal)
/// * `training filename`: Name of a file
/// * `test filename`:     Name of a file
/// # Optional
/// * `-t`: Change sigmoid activation function to tanh
/// * `-n integer`: Set number of hidden neurons (default: 3)
/// * `-i integer`: Set iterations for training (default: 5000)
/// * `-l float`: Set learning rate (dafault: 0.1)
///
/// # Return Value
/// ```io::Result<String>```
/// file content as String
///
/// # Remarks
/// Gets filename as a &str and opens the file.
/// The input of the file will be converted to a String.

fn main() {
    let matches = parse_args(); // clap crate: get terminal arguments

    // Terminal options -i, -n, -l, -t
    let iter = matches
        .value_of("iter")
        .map(|v| v.parse().or_exit("i is not a valid integer", 1))
        .unwrap_or(5000); // default iteration 5000

    let h_neurons = matches
        .value_of("hidden neurons")
        .map(|v| v.parse().or_exit("h is not a valid integer", 1))
        .unwrap_or(3); // default hidden neurons 3

    let learning_r = matches
        .value_of("learning rate")
        .map(|v| v.parse::<f32>().or_exit("l is not a valid float number", 1))
        .unwrap_or(0.1); // default learning rate 0.1

    let mut f: &Fn(Array1<f32>) -> Array1<f32> = &sig;
    let mut f_p: &Fn(Array1<f32>) -> Array1<f32> = &sig_prime;

    if matches.is_present("tanh") {
        f = &tanh;
        f_p = &tanh_prime;
    }

    // Read file and put into 2D vector
    let train_data: Vec<Vec<f32>> = reader::get_data(matches.value_of("TRAIN").unwrap());
    let test_data: Vec<Vec<f32>> = reader::get_data(matches.value_of("TEST").unwrap());

    // Get number of features aka input neurons
    // Create a new Settings object (weights, biases, ...)
    let feature_len = arr1(&train_data[0][1..]).len();
    let mut settings = nn::Settings::new(feature_len, h_neurons, learning_r);

    //  ------ TRAINING ------
    println!(
        "TRAINING \n iterations={} hidden_neurons={} learning_rate={}",
        iter, h_neurons, learning_r
    );
    for _x in 0..iter {
        train(&train_data, &mut settings, f, f_p);
        //print!("\r {}%             ", (x+1) as f32 / iter as f32 * 100.0); //Status only for non Windows
    }
    //  ------ EVALUATING  ------
    println!("\nEVALUATION");
    eval(&test_data, settings, f);
}
