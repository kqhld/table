import pandas as pd
from tablegpt.data_generator import DataGenerator # 从 tablegpt 模块中导入 DataGenerator 类
import argparse
import os# 导入 os 模块，用于处理文件和目录路径

if __name__ == "__main__":
    parser = argparse.ArgumentParser()#解析终端命令行
    #添加task参数
    parser.add_argument(
        "--task",
        default="TableSummary",
        choices=[
            "ColumnFinding",
            "MissingValueIdentification",
            "TableQuestion",
            "ColumnTypeAnnotation",
            "EntityMatching",
            "SchemaMatching",
            "DataImputation",
            "ErrorDetection",
            "ListExtraction",
            "HeaderValueMatching",
            "NL2SQL",
            "TableSummary",
            "ColumnAugmentation",
            "RowAugmentation",
            "RowColumnSwapping",
            "RowColumnFiltering",
            "RowColumnSorting",
            "Row2RowTransformation",
        ],
    )
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--source_dir", default="source")#源数据目录参数
    parser.add_argument("--save_dir", default="tablegpt_data")
    parser.add_argument("--num_test_fewshot_samples", default=3, type=int)
    parser.add_argument("--prob_train_fewshot", default=0.5, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--augment", default=False, action="store_true")
    parser.add_argument("--n_jobs", default=8, type=int)
    args = parser.parse_args()#解析终端命令行


    task_data_dir = os.path.join(args.source_dir, args.task)# 构建任务数据目录的路径，将源数据目录和任务名拼接起来
    train_data_dir = os.path.join(task_data_dir, "train")# 构建训练数据目录的路径，将任务数据目录和 "train" 拼接起来
    test_data_dir = os.path.join(task_data_dir, "test") # 构建测试数据目录的路径，将任务数据目录和 "test" 拼接起来

    print(type(train_data_dir))  # 检查数据类型
    print(train_data_dir[:3])  # 查看前 3 个数据
    print(f"Generating {args.mode} data for {args.task}")# 打印当前正在生成的任务和模式的信息

    if args.mode == "train":
        # 如果模式为训练，则创建一个 DataGenerator 对象，设置相应的参数
        data_generator = DataGenerator(
            args.task,
            mode="train",
            use_random_template=True,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            augment=args.augment,
        )
        # 调用 generate_data 方法生成训练数据
        data = data_generator.generate_data(train_data_dir, train_data_dir)
        # 构建保存文件名，根据模式和任务名，若有数据增强则添加 "_augment"
        save_name = f"{args.mode}_{args.task}"
        if args.augment:
            save_name += "_augment"

    else:
        test_data_generator = DataGenerator(
            # 如果模式为测试，则创建一个 DataGenerator 对象，设置相应的参数
            args.task,
            mode="test",
            use_random_template=False,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            num_test_fewshot_samples=args.num_test_fewshot_samples,
        )
        # 调用 generate_data 方法生成测试数据
        data = test_data_generator.generate_data(test_data_dir, train_data_dir)

        print(data[:3])  # 查看前 3 个数据
        save_name = f"{args.mode}_{args.task}"

        if args.num_test_fewshot_samples == 0:
            save_name += "_zeroshot"
        else:
            save_name += "_fewshot"

    if not os.path.exists(os.path.join(args.save_dir, args.mode)):
        os.makedirs(os.path.join(args.save_dir, args.mode))

    print(type(data))  # 检查数据类型
    print(data[:3])  # 查看前 3 个数据

    # data.to_json(
    #     os.path.join(
    #         args.save_dir,
    #         args.mode,
    #         f"{save_name}.jsonl",
    #     ),
    #     lines=True,
    #     orient="records",
    # )
    data.to_csv(
        os.path.join(
            args.save_dir,
            args.mode,
            f"{save_name}.csv",
        ),
        index=True  # 如果不需要保存行索引，可以设置 index=False
    )
