from aws_bedrock_agent.pipeline_runner import run_from_csv_with_text
from config import get_config

if __name__ == '__main__':
    cfg = get_config()
    run_from_csv_with_text(
        option_name='option1_llm',
        csv_in_path=cfg['data']['csv_option1'],
        csv_copy_out_path=cfg['data']['csv_option1'],
        json_out_path=cfg['data']['json_option1']
    )
    run_from_csv_with_text(
        option_name='option2_llm',
        csv_in_path=cfg['data']['csv_option2'],
        csv_copy_out_path=cfg['data']['csv_option2'],
        json_out_path=cfg['data']['json_option2']
    )
    print('Completed both options')
