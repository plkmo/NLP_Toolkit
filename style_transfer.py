
from nlptoolkit.style_transfer.add_misc.misc import save_as_pickle, Config
from nlptoolkit.style_transfer.data import load_dataset
from nlptoolkit.style_transfer.models import StyleTransformer, Discriminator
from nlptoolkit.style_transfer.train import train
from nlptoolkit.style_transfer.infer import infer_from_trained

from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def main(config):
    save_as_pickle('style_transfer/config.pkl', config)
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)
    
    train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/style_transfer/yelp/',\
                        help="Full path to style-transfer dataset")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of style transfer classes")
    parser.add_argument("--max_features_length", type=int, default=16, help="Max length of features")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer model dimension")
    parser.add_argument("--num", type=int, default=4, help="Transformer number of layers per block")
    parser.add_argument("--n_heads", type=int, default=4, help="Transformer number of attention heads")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr_F", type=float, default=0.0001, help="Generator learning rate")
    parser.add_argument("--lr_D", type=float, default=0.0001, help="Discriminator learning rate")
    parser.add_argument("--num_iters", type=int, default=2000, help="No of training iterations")
    parser.add_argument("--save_iters", type=int, default=50, help="No of iterations per checkpoint saving")
    
    parser.add_argument("--train", type=int, default=1, help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=1, help="Infer input sentence labels from trained model")
    parser.add_argument("--checkpoint_Fpath", type=str, default='./data/style_transfer/Jan18190541/ckpts/900_F.pth',\
                        help="Full path to style-transfer F checkpoint (for inference)")
    parser.add_argument("--checkpoint_Dpath", type=str, default='./data/style_transfer/Jan18190541/ckpts/900_D.pth',\
                        help="Full path to style-transfer D checkpoint (for inference)")
    
    args = parser.parse_args()
    config = Config(args)
    
    if args.train == 1:
        main(config)
    
    if args.infer == 1:
        inferer = infer_from_trained(F_path=args.checkpoint_Fpath, \
                                 D_path=args.checkpoint_Dpath,\
                                 config_file='style_transfer/config.pkl', generator_only=True)
    
        gen_sent = inferer.infer_sentence(sent='The food here is really good.', target_style=0)
        print('Style-transferred sentence: ', gen_sent)