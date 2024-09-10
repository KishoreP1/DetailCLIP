import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint for DetailCLIP')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--output', type=str, help='destination file name')
    parser.add_argument("--use_teacher", action='store_true', help='extract teacher vit or student vit')
    args = parser.parse_args()
    return args

def extract(get_teacher, ck):
    new_ck = {}
    look_for = 'module.visual_ema.' if get_teacher else 'module.visual.'
    print('looking for: ', look_for)
    for k, v in ck['state_dict'].items():
        if k.startswith(look_for):
            new_ck[k.replace(look_for, '')] = v
    return new_ck

def main():
    args = parse_args()
    print(args)
    ck = torch.load(args.checkpoint, map_location='cpu')
    extracted = extract(args.use_teacher, ck)
    print('extracted keys: ', extracted.keys())
    torch.save(extracted, args.output)
    print('done')

if __name__ == '__main__':
    main()