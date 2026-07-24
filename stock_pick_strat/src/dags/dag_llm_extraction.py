from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime.today(),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=15),
    "max_active_tasks": 40,
}

dag = DAG(
    dag_id="data_llm_extraction",
    default_args=default_args,
    description="A DAG to extract features from crawled items",
    schedule_interval=timedelta(days=7),
)

# Define the paths for the CLI and the commands
cli_path = "/opt/airflow/"

nbr_threads_llm = Variable.get("threads_details", default_var=20)
apis = Variable.get("apis", default_var="open_ai")


# garbage collector first  then crawl
step_gpt_reformulate = BashOperator(
    cwd=cli_path,
    task_id=f"step_gpt_reformulate",
    bash_command=f"python -m src gpt_extraction step-inference-gpt -t {nbr_threads_llm} --gpt-methode {apis} --object reformulate",
    dag=dag,
)

step_gpt_reformulate_second_pass = BashOperator(
    cwd=cli_path,
    task_id=f"step_gpt_reformulate_second_pass",
    bash_command=f"python -m src gpt_extraction step-inference-gpt -t {nbr_threads_llm} --gpt-methode {apis} --object reformulate",
    dag=dag,
)

step_gpt_clean_reformulate = BashOperator(
    cwd=cli_path,
    task_id=f"step_gpt_clean_reformulate",
    bash_command="python -m src dataclean step-cleaning-gpt-object-category",
    dag=dag,
)

step_gpt_clean_reformulate_second_pass = BashOperator(
    cwd=cli_path,
    task_id=f"step_gpt_clean_reformulate_second_pass",
    bash_command="python -m src dataclean step-cleaning-gpt-object-category",
    dag=dag,
)

step_gpt_feature_extract = BashOperator(
    cwd=cli_path,
    task_id=f"step_gpt_feature_extraction",
    bash_command=f"python -m src gpt_extraction step-inference-gpt -t {nbr_threads_llm} --gpt-methode {apis} --object painting",
    dag=dag,
)

step_gpt_clean_extraction = BashOperator(
    cwd=cli_path,
    task_id=f"step_gpt_clean_extraction",
    bash_command="python -m src dataclean step-cleaning-gpt-feature-extraction",
    dag=dag,
)

step_remove_gpt = BashOperator(
    cwd=cli_path,
    task_id=f"step_remove_gpt",
    bash_command=f"python -m src dataclean step-remove-gpt",
    dag=dag,
)

step_index_reformulate_english = BashOperator(
    cwd=cli_path,
    task_id=f"step_text_indexation_en",
    bash_command=f"python -m src dataclean step-text-indexation --language english",
    dag=dag,
)

step_index_reformulate_french = BashOperator(
    cwd=cli_path,
    task_id=f"step_text_indexation_fr",
    bash_command=f"python -m src dataclean step-text-indexation --language french",
    dag=dag,
)

# Full dag
(
    step_gpt_reformulate
    >> step_gpt_clean_reformulate
    >> step_remove_gpt
    >> step_gpt_reformulate_second_pass
    >> step_gpt_clean_reformulate_second_pass
    >> step_index_reformulate_english
    >> step_index_reformulate_french
    >> step_gpt_feature_extract
    >> step_gpt_clean_extraction
)
