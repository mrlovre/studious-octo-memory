import dataset as ds
import recurrent_neural_network as rnn

def main(path='data/star_trek.txt'):
    dataset = ds.Dataset(path)
    rnn.run_language_model(dataset, save_file='train_data/star_trek', max_epochs=10000)

if __name__ == '__main__':
    main()
