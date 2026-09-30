import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Local CPU inference experiments")
    parser.add_argument("command", choices=["setup", "export", "parity", "quantize", "evaluate", "compare", "benchmark", "worker", "demo", "report"])
    parser.add_argument("--variant", choices=["pytorch_fp32", "onnx_fp32", "onnx_int8"], default="onnx_int8")
    parser.add_argument("--subset", choices=["development", "final"], default="development")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--destination")
    parser.add_argument("--text", default="I really enjoyed this film.")
    args = parser.parse_args()
    if args.command != "setup":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command in ("setup", "export", "parity", "quantize"):
        from quantbench import pipeline
        getattr(pipeline, args.command)()
    else:
        from quantbench import experiment
        if args.command == "report":
            from quantbench.report import generate
            generate()
        elif args.command == "evaluate":
            experiment.evaluate(args.variant, args.subset)
        elif args.command == "compare":
            experiment.paired_quality(args.subset)
        elif args.command == "benchmark":
            experiment.benchmark()
        elif args.command == "worker":
            if not args.destination:
                parser.error("worker requires --destination")
            experiment.worker(args.variant, args.repetition, args.destination)
        else:
            import json
            import numpy as np
            from quantbench.runners.local import LocalRunner
            runner = LocalRunner(args.variant)
            runner.load()
            logits = runner.predict(runner.prepare(runner.preprocess([args.text])))[0]
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
            print(json.dumps({"label": ["NEGATIVE", "POSITIVE"][int(logits.argmax())],
                              "scores": probabilities.tolist(), "variant": args.variant,
                              "model_revision": runner.manifest["model_revision"]}))


if __name__ == "__main__":
    main()
