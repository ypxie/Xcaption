import bladder
import flickr8k
import coco
import flickr30k

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
            'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
            'coco': (coco.load_data, coco.prepare_data),
            'bladder': (bladder.load_data, bladder.prepare_data)}

    
