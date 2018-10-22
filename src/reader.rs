//#############################################################################
// Author:      Le Duyen Sandra Vu
// Matr.Nr.:    4171456
// Compiler:    rustc 1.27.0 (3eda71b00 2018-06-19)
// Cargo:       1.27.0 (1e95190e5 2018-05-27)
// OS:          MacOS 10.13.4
// Subject:     Single hidden layer neural network for classification
//
// READER.VERSION 29.08.18
//#############################################################################

/*!
 * has one public function get_data(filename: &str)
 * to convert file content into [line number][column1 value, column2 value, ...].
 */

use std::fs::File;
use std::io::{self, Read};
use stdinout::OrExit;

/// # Arguments
/// * `filename`: Name of a file
///
/// # Return Value
/// ```io::Result<String>```
/// file content as String
///
/// # Remarks
/// Gets filename as a &str and opens the file.
/// The input of the file will be converted to a String.
fn file_to_string(s: &str) -> io::Result<String> {
    let mut file = File::open(s).or_exit("Cannot open file", 1);
    let mut s = String::new();
    file.read_to_string(&mut s)?;

    if s == "" {
        panic!("Empty file");
    }
    Ok(s)
}

/// # Arguments
/// * `file`: &str
///
/// # Return Value
/// ```Vec<Vec<f32>>```
/// [line number][column1 value, column2 value, ...]
///
/// # Remarks
/// Gets a &str and converts its content to a 2D vector.
fn data_to_vec<'a>(file: &'a str) -> Vec<Vec<f32>> {
    file.lines()
        .map(|line| {
            line.split_whitespace()
                .map(|s| {
                    s.parse::<f32>()
                        .expect("File content has to contain only numbers.")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

/// # Arguments
/// * `filename`: Name of a file
///
/// # Return Value
/// ```Vec<Vec<f32>>```
/// [line number][column1 value, column2 value, ...]
///
/// # Remarks
/// Gets filename as a &str and converts the content of the file
/// first to a String and then to a 2D vector.
pub fn get_data(filename: &str) -> Vec<Vec<f32>> {
    let file = file_to_string(filename).unwrap();

    if data_to_vec(&file)[0].len() < 3 {
        panic!("Not enough features. 1 label and at least 2 features necessary.");
    }

    data_to_vec(&file)
}
