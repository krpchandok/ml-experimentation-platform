BRAND = "mlplat bakehouse"
TAGLINE = "See where your training time goes, and where to bake it next."

PAGE_TITLES = {
    "home": "Where should I train?",
    "story": "Run story",
    "compare": "Before and after",
    "runs": "All bakes",
}

HOME_INTRO = (
    "Training a model is like running a bakery. Bakers prepare trays of dough, and the oven bakes them. "
    "We timed a short bake on your machine. Here is how long the full bake would take in each kitchen, "
    "and what it would cost."
)
HOME_EMPTY = "No bakes yet. Time a short one with: mlplat plan --name my-model --total-steps 20000 -- python train.py"
HOME_PICK_RUN = "Which timed bake should we plan from?"
HOME_TOTAL_STEPS = "How many trays in the full bake?"
HOME_TOTAL_STEPS_HELP = "A tray is one training step. Use the number of steps your full training will run."
HOME_PREFER = "What matters most?"
HOME_PREFER_OPTIONS = {"balanced": "Balanced", "time": "Fastest", "cost": "Cheapest"}
HOME_ESTIMATE_NOTE = "All times and costs are estimates from a short timed bake."
HOME_PLACEHOLDER_NOTE = "Some kitchens still use placeholder numbers. Edit targets.yaml with real specs and prices."
HOME_NO_PLAN = "This bake can't be planned yet: {reason}"
HOME_TABLE = "See every kitchen as a table"

FIX_TITLE = "Fix this first"
FIX_BODY = ("Your oven waits for dough most of the time. More bakers would help far more than a bigger oven, "
            "and it's free.")
FIX_STILL_BOUND = "Even then, the bakers stay the slowest part. Next, make the dough simpler or prepare it once."
FIX_OPTION_FIX = "Add bakers ({before} → {after})"
FIX_OPTION_UPGRADE = "Rent a faster oven ({name})"

BEST_BADGE = "Best choice"
CARD_TIME = "Full bake"
CARD_COST = "Cost"
CARD_PLACEHOLDER = "Placeholder numbers"
CARD_FIT_ONE = "Fits in one session"
CARD_FIT_MANY = "Needs {sessions} sessions: save your progress between them"
CARD_FIT_WEEK = "Takes about {weeks} weeks of your weekly hours"
CARD_NO_ROOM = "Not enough counter space for this recipe"
CARD_NO_OVEN_ROOM = "The oven is too small for this recipe"
CARD_BAKERS = "with {bakers}"
CARD_LIMIT_BAKERS = "Bakers are the slowest part"
CARD_LIMIT_OVEN = "The oven is the slowest part"

BAKERIES = {
    "home_kitchen": "Your home kitchen",
    "community_kitchen": "Community kitchen",
    "market_stall": "Market stall",
    "rented_bakery": "Rented bakery",
}

STAGE_NAMES = {
    "pantry": "Ingredients",
    "bakers": "Bakers",
    "oven": "Oven",
    "goods": "Baked trays",
}
STAGE_DETAILS = {
    "pantry": "Data read from disk",
    "bakers": "Preparing dough",
    "oven": "Baking",
    "goods": "Training steps done",
}

VERDICT_PLAIN = {
    "preprocessing_bound": ("The bakers can't keep up", "The oven keeps waiting for the next tray of dough."),
    "main_process_bound": ("The head baker is the slowest part",
                           "The dough is ready, but the person loading the oven can't go any faster."),
    "io_bound": ("Ingredients arrive slowly", "The bakers wait for ingredients from the pantry."),
    "memory_pressure": ("The counter is almost full", "There is barely room left to work."),
    "gpu_not_used": ("The oven stayed cold", "Nothing was baked in the oven. The model ran on the work stations."),
    "healthy": ("The oven stays busy", "Everything is flowing. The oven is the busiest part, as it should be."),
    "underutilized": ("Everyone is waiting around", "Neither the oven nor the bakers were busy."),
    "insufficient_data": ("Not enough baking to judge", "This bake was too short to tell what happened."),
}

PLAIN_TRY = {
    "preprocessing_bound": ["Hire more bakers: use more data-loader workers.",
                            "Make the dough simpler, or prepare it once and reuse it.",
                            "Let the oven do some of the prep work."],
    "main_process_bound": ["Load bigger trays so each trip into the oven does more.",
                           "Stop checking on the oven after every single tray."],
    "io_bound": ["Keep ingredients closer: put the data on a faster local disk.",
                 "Prepare the ingredients once and keep them on the counter."],
    "memory_pressure": ["Use smaller trays.", "Use fewer bakers, or let them share ingredients."],
    "gpu_not_used": ["Make sure the recipe actually uses the oven."],
    "healthy": ["Nothing to fix in the kitchen. To go faster you need a faster oven or a lighter recipe."],
    "underutilized": ["Look for waiting that isn't baking, like downloads or saving too often.",
                      "Try bigger trays."],
    "insufficient_data": ["Bake for longer so there is enough to measure."],
}

STORY_SENTENCES = {
    "preprocessing_bound": "Your oven sat empty for {idle} of {span}, waiting for dough.",
    "main_process_bound": "Your bakers kept up, but the head baker loading the oven was the slow part. "
                          "The oven sat empty for {idle} of {span}.",
    "io_bound": "Your bakers spent most of {span} waiting for ingredients from the pantry.",
    "memory_pressure": "Your counter was almost full: {peak} used out of {limit}.",
    "gpu_not_used": "Your oven stayed cold for the whole {span}. The baking happened on the work stations instead.",
    "healthy": "Your oven was busy {busy} of the time. Nice baking.",
    "underutilized": "Nobody was very busy during these {span}. The oven was busy {busy} of the time.",
    "insufficient_data": "This bake was too short to tell what happened.",
}

SLOWEST = {
    "preprocessing_bound": "Slowest step: your {workers}, busy {worker_busy} of the time",
    "main_process_bound": "Slowest step: the head baker loading the oven",
    "io_bound": "Slowest step: fetching ingredients from the pantry",
    "memory_pressure": "Watch out: the counter is nearly full",
    "gpu_not_used": "Slowest step: baking without the oven",
    "healthy": "Busiest step: the oven, as it should be",
    "underutilized": "Nothing is the clear bottleneck",
    "insufficient_data": "Not enough data to find the slowest step",
}
SLOWEST_STAGE = {
    "preprocessing_bound": "bakers", "main_process_bound": "oven", "io_bound": "pantry",
    "memory_pressure": "bakers", "gpu_not_used": "oven", "healthy": "oven", "underutilized": None,
    "insufficient_data": None,
}

STORY_CARDS = {
    "idle": ("Oven sat empty", "Time the oven waited with nothing inside."),
    "busy": ("Oven busy", "Share of the time the oven was baking."),
    "tray_time": ("Time per tray", "How long one training step took."),
    "rate": ("Trays per second", "How many training steps finished each second."),
    "bakers": ("Bakers busy", "How hard each data-loader worker was working."),
    "counter": ("Counter space used", "Most memory the run used at once."),
}

CHART_TITLES = {
    "cpu": ("How hard each baker worked", "CPU per process (% of one core)"),
    "memory": ("Counter space used", "Resident memory, MiB (RSS)"),
    "io": ("Ingredients fetched from the pantry", "Storage I/O, MiB/s"),
    "gpu": ("How busy the oven was", "GPU utilization, %"),
    "gpu_memory": ("Oven space used", "GPU memory used, MiB"),
    "step_time": ("Time per tray", "Step time, ms"),
}

PROCESS_LABELS = {"main": "head baker", "worker": "baker", "other": "other helpers"}

TECH_TOGGLE = "Show technical details"
TECH_TOGGLE_HELP = "Adds the engineering terms and raw numbers behind each plain-language sentence."
TECH_HEADING = "Technical details"
WHAT_TO_TRY = "What to try"
ANIMATION_NOTE = "Slowed down {factor} times so you can see it."
REDUCED_MOTION_NOTE = "Animation is off because your device asks for reduced motion."
ESTIMATE_LABEL = "Estimate"

COMPARE_INTRO = "Pick a bake from before a change and one from after. Here is what changed."
COMPARE_BEFORE = "Before"
COMPARE_AFTER = "After"
COMPARE_FASTER = "{factor:.1f}× faster"
COMPARE_SLOWER = "{factor:.1f}× slower"
COMPARE_SAME = "About the same"
COMPARE_DETAILS = "Detailed comparison"

RUNS_INTRO = "Every bake you've timed, newest first."
RUNS_EMPTY = "No bakes yet. Start one with: mlplat run --name my-model -- python train.py"
RUNS_OPEN = "Open story"

STATUS = {
    "completed": ("Finished", ":material/check_circle:"),
    "profiled": ("Timed", ":material/timer:"),
    "running": ("Baking now", ":material/progress_activity:"),
    "failed": ("Failed", ":material/error:"),
    "interrupted": ("Stopped", ":material/pause_circle:"),
}

SEVERITY = {
    "critical": ("Needs attention now", ":material/dangerous:", "error"),
    "bottleneck": ("Slowing you down", ":material/speed:", "warning"),
    "warning": ("Worth a look", ":material/warning:", "warning"),
    "ok": ("Looking good", ":material/check_circle:", "success"),
    "info": ("For your information", ":material/info:", "info"),
    "unknown": ("Not sure yet", ":material/help:", "info"),
}

GLOSSARY = {
    "oven": "The GPU: the chip that does the heavy math of training.",
    "baker": "A data-loader worker: a helper process that prepares the next batch of training data.",
    "head baker": "The main training process: it loads each tray into the oven and runs the training loop.",
    "tray": "A training batch: the group of examples the model learns from in one step.",
    "tray size": "Batch size: how many examples fit in one training step.",
    "baked tray": "A training step: one batch processed by the model.",
    "oven sat empty": "Data wait: time the GPU spent idle, waiting for the next batch.",
    "counter space": "Memory (RAM): how much working space the run needs.",
    "work station": "A CPU core: one place where a baker can work.",
    "kitchen": "A training target: a place you can run training, like your own GPU, Colab, Kaggle or a cloud GPU.",
    "pantry": "Storage: where the training data lives before it is prepared.",
}

TECH_TERMS = {
    "bakers": "DataLoader workers (child processes, CPU from /proc/<pid>/stat)",
    "oven": "GPU (NVML utilization)",
    "head baker": "main training process",
    "counter": "resident set size (RSS, /proc/<pid>/status VmRSS)",
    "pantry": "storage reads (/proc/<pid>/io read_bytes)",
}


def duration(seconds):
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.0f} seconds" if seconds >= 10 else f"{seconds:.1f} seconds"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    hours = int(minutes // 60)
    return f"{hours} h {int(minutes - hours * 60):02d} min"


def short_duration(seconds):
    if seconds is None:
        return "-"
    milliseconds = seconds * 1000
    if round(milliseconds, 1) < 100:
        return f"{milliseconds:.1f} ms"
    if round(milliseconds) < 1000:
        return f"{milliseconds:.0f} ms"
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = int(minutes // 60)
    return f"{hours} h {int(minutes - hours * 60):02d} min"


def percent(fraction):
    return "-" if fraction is None else f"{fraction * 100:.0f}%"


def cost(value):
    if not value:
        return "Free"
    return "Under $0.01" if value < 0.01 else f"${value:,.2f}"


def busy(fraction, technical=False):
    if fraction is None:
        return "-"
    return f"{fraction * 100:.0f}%" if technical else f"{min(fraction, 1.0) * 100:.0f}%"


def bakers(count):
    return "no bakers" if count == 0 else ("1 baker" if count == 1 else f"{count} bakers")


def verdict_plain(verdict_id):
    return VERDICT_PLAIN.get(verdict_id, VERDICT_PLAIN["insufficient_data"])


def story_sentence(verdict_id, idle=None, span=None, busy=None, peak=None, limit=None):
    template = STORY_SENTENCES.get(verdict_id, STORY_SENTENCES["insufficient_data"])
    values = {"idle": duration(idle), "span": duration(span), "busy": percent(busy), "peak": peak or "-",
              "limit": limit or "-"}
    if "{idle}" in template and idle is None:
        template = VERDICT_PLAIN.get(verdict_id, VERDICT_PLAIN["insufficient_data"])[1]
    return template.format(**values)


def slowest_label(verdict_id, workers=0, worker_busy=None):
    template = SLOWEST.get(verdict_id, SLOWEST["insufficient_data"])
    return template.format(workers=bakers(workers), worker_busy=busy(worker_busy))
