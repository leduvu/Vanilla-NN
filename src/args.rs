//#############################################################################
// Author:      Le Duyen Sandra Vu
// Matr.Nr.:    4171456
// Compiler:    rustc 1.27.0 (3eda71b00 2018-06-19)
// Cargo:       1.27.0 (1e95190e5 2018-05-27)
// OS:          MacOS 10.13.4
// Subject:     Single hidden layer neural network for classification
//
// ARGS.VERSION 29.08.18
//#############################################################################

/*!
 * deals with the terminal arguments and possible options.
 * short: -i long: --iter
 */

use clap::{App, Arg, ArgMatches};

pub fn parse_args() -> ArgMatches<'static> {
    App::new("neural_network")
        .version("0.1.0")
        .author("Le Duyen Sandra Vu <leduvu@gmail.com>")
        .usage(
            "neural_network <Training FILE> <Test FILE> [OPTIONS]\n\n\
           EXAMPLES:\n\
           neural_network xor.txt xor.txt -t -i 2000 -h 2 -l 0.5\n\
           neural_network -t -i 2000 -h 2 -l 0.5 xor.txt xor.txt\n\
           neural_network xor.txt xor.txt\n\n\

           OPTIONS:\n\
           -t, --tanh       Change sigmoid activation function to tanh\n\
           -i, --iter       Set iterations for training (default: 5000)\n\
           -n, --hidden     Set number of hidden neurons (default: 3)\n
           -l, --learn      Set learning rate (dafault: 0.1)",
        )
        .help(
            "Neural Network v0.1.0\n\
           1 layer feedforward NN\n\
           (C) leduvu@mail.com\n\n\

           USAGE:   neural_network <Training FILE> <Test FILE> [OPTIONS]\n\n\
           EXAMPLE: neural_network xor.txt xor.txt -t -i 2000 -h 2 -l 0.5

           OPTIONS:\n\
           -t, --tanh       Change sigmoid activation function to tanh\n\
           -i, --iter       Set iterations for training (default: 5000)\n\
           -n, --hidden     Set number of hidden neurons (default: 3)\n\
           -l, --learn      Set learning rate (default: 0.1)\n\
           -h, --help       Dispay this message\n\
           -V, --version    Display version info\n",
        )
        .arg(
            Arg::with_name("iter")
                .short("i")
                .long("iter")
                .help("Number of iterations: (default: 5000)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("hidden neurons")
                .short("n")
                .long("hidden")
                .help("Number of hidden neurons: (default: 3)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("l")
                .short("l")
                .long("learn")
                .help("Learning rate: (default: 0.1)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("tanh")
                .short("t")
                .long("tanh")
                .help("Tanh activationfunction")
                .takes_value(false),
        )
        .arg(
            Arg::with_name("TRAIN")
                .help("Train data")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name("TEST")
                .help("Test data")
                .index(2)
                .required(true),
        )
        .get_matches()
}
