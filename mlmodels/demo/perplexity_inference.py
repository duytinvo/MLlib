from mlmodels.inference.perplexity_calculator import main
from mlmodels.utils.csvIO import CSV

if __name__ == '__main__':
    file = '../../../media/data/coco/paraphrase-coco.csv'
    data = CSV.read(file)
    flat_list = [item for sublist in data for item in sublist]
    avg_perplexity = main(flat_list)
    print(avg_perplexity, flush=True)