def HM(iid_acc, ood_acc):
    a = 2*iid_acc*ood_acc
    b = iid_acc + ood_acc
    c = a/b
    return c
    
if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='HM')
    parser.add_argument('--iid_acc', type=float)
    parser.add_argument('--ood_acc', type=float)
    
    args = parser.parse_args()
    
    hm_acc = HM(args.iid_acc, args.ood_acc)
    
    print("hm_acc", hm_acc)