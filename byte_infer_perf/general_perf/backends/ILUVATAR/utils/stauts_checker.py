def check_status(result_dict, args):
    is_valid = True
    if args.acc_target is not None:
        acc_result = result_dict["acc_result"]
        if acc_result < args.acc_target:
            print(f"Expected acc_target is {args.acc_target}, got {acc_result}")
            is_valid = False
            
    if args.fps_target is not None:
        fps_result = result_dict["fps_result"]
        if fps_result < args.fps_target:
            print(f"Expected fps_target is {args.fps_target}, got {fps_result}")
            is_valid = False
    
    if is_valid:
        print("\n====Test Success!====\n")
    else:
        print("\n====Test failed!====\n")
        exit(1)
    
    