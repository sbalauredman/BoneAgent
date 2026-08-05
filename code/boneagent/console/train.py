from __future__ import annotations

import logging
from pathlib import Path

import torch

from boneagent.console.common import configure_logging, load_yaml, parser
from boneagent.data.tabular import MaterialDataset, read_tabular_records
from boneagent.engine.training import (
    CosineScheduler,
    PropertyTrainer,
    TrainerState,
    atomic_checkpoint,
)
from boneagent.randomness import set_seed
from boneagent.twin.tabular import ScaffoldPropertyNetwork

logger = logging.getLogger(__name__)


def main() -> None:
    argument_parser = parser("Train a BoneAgent property surrogate")
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    arguments = argument_parser.parse_args()
    configure_logging(arguments.verbose)
    config = load_yaml(arguments.config)
    training = config["training"]
    optimizer_config = config["optimizer"]
    seed = int(training.get("seeds", [42])[0])
    set_seed(seed)
    feature_columns = [
        "calcium_phosphorus_ratio",
        "hydroxyapatite_fraction",
        "beta_tcp_fraction",
        "dopant_atomic_number",
        "dopant_fraction",
        "porosity",
        "pore_diameter_micrometer",
        "strut_diameter_micrometer",
        "interconnectivity",
        "sintering_temperature_celsius",
        "oxygen_fraction",
        "holding_hours",
    ]
    target_columns = [
        "compressive_strength_mpa",
        "elastic_modulus_gpa",
        "degradation_weeks",
        "cell_viability",
        "osteogenic_potential",
        "clinical_survival",
        "bone_ingrowth_fraction",
        "formation_energy_ev_atom",
    ]
    records = read_tabular_records(arguments.data, feature_columns, target_columns)
    split_index = max(1, int(0.85 * len(records)))
    training_data = MaterialDataset(records[:split_index])
    validation_data = MaterialDataset(records[split_index:])
    batch_size = int(training["batch_size"])
    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScaffoldPropertyNetwork(output_channels=len(target_columns)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    epochs = int(training["epochs"])
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = CosineScheduler(optimizer, total_steps)
    trainer = PropertyTrainer(
        model,
        optimizer,
        scheduler,
        device,
        float(training["gradient_clip"]),
        str(training["precision"]),
        int(training["gradient_accumulation"]),
    )
    best = float("inf")
    patience = int(training.get("early_stopping", {}).get("patience", epochs))
    stale = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        training_loss = trainer.train_epoch(train_loader)
        validation_loss = trainer.validate(validation_loader)
        global_step += len(train_loader)
        if validation_loss < best:
            best = validation_loss
            stale = 0
            state = TrainerState(epoch, global_step, best, stale, seed)
            atomic_checkpoint(arguments.output, model, optimizer, scheduler, state)
        else:
            stale += 1
        logger.info(
            "epoch=%d training_loss=%.7f validation_loss=%.7f best=%.7f",
            epoch,
            training_loss,
            validation_loss,
            best,
        )
        if stale >= patience:
            break


if __name__ == "__main__":
    main()
