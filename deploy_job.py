import argparse
import linear_regression_final as linreg
import logistic_regression_final as logreg
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help="Task to be done", choices=['linear_reg', 'logistic_reg'])
parser.add_argument("dataset_name", type=str, help="Filename of the dataset stored in \'data_tum_base\'")
parser.add_argument("test_dset", type=str, help="Filename of the test set stored in \'data_tum_base\'")
parser.add_argument("no_of_nodes", type=int, help="Number of nodes")
parser.add_argument("no_of_iter", type=int, help="Number of iterations")
parser.add_argument("lrate", type=float, help="Learning rate")
parser.add_argument("strag", type=float, help="Stragglers mentioned in terms of seconds(algo = 'time_strag') or percentage(algo = 'percent_strag' or 'ErasHead' or 'StocGD')")
parser.add_argument("choice", type=str, help="Choice of assignment matrix", choices=['partition', 'FRC', 'StocGD'])
parser.add_argument("factor", type=float, help="Multiplicative factor for the randomly initialized weight.")
parser.add_argument("algo", type=str, help="Algorithm being used", choices=['time_strag', 'percent_strag', 'ErasHead', 'StocGD'])
parser.add_argument("c", type=int, help="c value used in Erasure Head algorithm")
parser.add_argument("d", type=int, help="Redundancy value(d) used in Stochastic GC")
parser.add_argument("--if_ones", type=int, help="If columns of ones need to be added to dataset", choices=[0, 1], default=0)
parser.add_argument("--connections", type=int, help="Maximum number of workers to be used for concurrent node calls", default=200)
parser.add_argument("--timeout", type=float, help="Timeout for http calls", default=180)
parser.add_argument("--if_custom_init_wt", action='store_true', help="Use custom initial weights stored as initial_wt.csv? add flag if you want True")
parser.add_argument("--if_change_assgn", action='store_true', help="Use varying assignment matrices? add flag if you want True")
parser.add_argument("--if_verbose", action='store_true', help="Be verbose in console printing? add flag if you want True")
parser.add_argument("--if_variable_lrate", action='store_true', help="Use decreasing learning rate computed by compute_alpha()? add flag if you want True")
#parser.add_argument("--if_test", action='store_true', help="Test the obtained weights? add flag if you want True")

args = parser.parse_args()

cwd = os.path.abspath(os.getcwd())

if args.task == 'linear_reg':
	folder_name = linreg.master(args.dataset_name, args.test_dset, args.no_of_nodes, args.no_of_iter, args.lrate, args.strag, args.choice, args.c, args.factor, args.connections, args.timeout, args.algo, args.d, args.if_ones, args.if_custom_init_wt, args.if_change_assgn, args.if_verbose, args.if_variable_lrate)
	
	# Moving the log file into the folder meant for this run
	path = os.path.join(cwd, folder_name)
	os.replace(os.path.join(cwd, 'Messages.log'), os.path.join(path, 'Messages.log'))

elif args.task == 'logistic_reg':
	folder_name = logreg.master(args.dataset_name, args.test_dset, args.no_of_nodes, args.no_of_iter, args.lrate, args.strag, args.choice, args.c, args.factor, args.connections, args.timeout, args.algo, args.d, args.if_ones, args.if_custom_init_wt, args.if_change_assgn, args.if_verbose, args.if_variable_lrate)

	# Moving the log file into the folder meant for this run
	path = os.path.join(cwd, folder_name)
	os.replace(os.path.join(cwd, 'Messages.log'), os.path.join(path, 'Messages.log'))

else:
	print('Please give a task among the 2 choices. Exiting...')
	sys.exit()


#master(args.dataset_name, args.no_of_nodes, args.no_of_iter, args.lrate, args.strag, args.choice, args.c, args.factor, args.connections, args.timeout, args.algo, args.if_ones, args.if_custom_init_wt, args.if_change_assgn, args.if_verbose)
#master(dataset_name, no_of_nodes, no_of_iter, lrate, strag, choice, c, factor, connections, timeout, algo, if_ones, if_custom_init_wt, if_change_assgn, if_verbose):
