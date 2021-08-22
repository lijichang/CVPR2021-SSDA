import os
def folder_preparation(nowtime, args):
    main_path = 'record/%s_%s_%s_%s' % (nowtime, args.dataset, args.method, args.remark)

    if not os.path.exists(main_path):
        os.makedirs(main_path)
    # results saving
    results_file = os.path.join(main_path,
                                'results_%s_net_%s_%s_to_%s_num_%s' %
                                (args.method, args.net, args.source,
                                 args.target, args.num))

    # logs saving
    logs_file = os.path.join(main_path,
                             'logs_%s_net_%s_%s_to_%s_num_%s' %
                             (args.method, args.net, args.source,
                              args.target, args.num))

    # checkpath saving
    checkpath = os.path.join(main_path, 'checkpath',
                             '%s_net_%s_%s_to_%s_num_%s' %
                             (args.method, args.net, args.source,
                              args.target, args.num))
    if not os.path.exists(checkpath):
        os.makedirs(checkpath)

    return main_path, results_file, logs_file, checkpath
