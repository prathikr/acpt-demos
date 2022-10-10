import argparse
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import AzureCliCredential


def run_config_to_args(run_config):
    mapping = {
        "no_acc": ["--fp16", "True"],
        "ds": ["--fp16", "True", "--deepspeed", "True"],
        "ort": ["--fp16", "True", "--ort", "True"],
        "ds_ort": ["--fp16", "True", "--deepspeed", "True", "--ort", "True"],
    }
    return mapping[run_config]


def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="GPT2 Finetune AML job submission")

    # workspace
    parser.add_argument(
        "--ws_config",
        type=str,
        required=True,
        help="Workspace configuration. Path is absolute or relative to where script is called from",
    )
    parser.add_argument("--compute", type=str, required=True, help="Compute target to run job on")

    # distributed training config
    parser.add_argument("--nnode", type=int, default=1, help="No of nodes. Default is 1")
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="No of GPUs per node. Default is 8",
    )

    # accelerator hyperparameters
    parser.add_argument(
        "--run_config", choices=["no_acc", "ort", "ds", "ds_ort"], default="no_acc", help="Configs to run for model"
    )

    # parse args, extra_args used for job configuration
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    args = get_args(raw_args)
    run_config_args = run_config_to_args(args.run_config)

    root_dir = Path(__file__).resolve().parent
    component_dir = root_dir / "components"

    # connect to the workspace
    ws_config_path = root_dir / args.ws_config
    ml_client = MLClient.from_config(credential=AzureCliCredential(), path=ws_config_path)

    # code directory
    code_dir = component_dir / "finetune-code"
    environment_dir = component_dir / "environment"

    # tags
    tags = {
        "__run_config": args.run_config,
    }

    # define the command
    command_job = command(
        description="ACPT GPT2 Finetune Demo",
        display_name=f"gpt-finetune-{args.nnode}-{args.nproc_per_node}-{args.run_config}-{args.batch_size}",
        experiment_name="acpt-gpt2-finetune-demo",
        code=code_dir,
        command=(
            "python finetune_QA.py"
            f" {' '.join(run_config_args)}"
        ),
        environment=Environment(
            description="ACPT GPT2 fine-tune environment", build=BuildContext(path=environment_dir)
        ),
        distribution={
            "type": "pytorch",
            "process_count_per_instance": args.nproc_per_node,
        },
        compute=args.compute,
        instance_count=args.nnode,
        tags=tags,
    )

    # submit the command
    print("submitting job")
    returned_job = ml_client.jobs.create_or_update(command_job)
    print("submitted job")

    aml_url = returned_job.studio_url
    print("job link:", aml_url)


if __name__ == "__main__":
    main()
