import os
import sqlite3

from difflib import SequenceMatcher
from func_timeout import func_set_timeout, FunctionTimedOut
from sql_metadata import Parser

def find_most_similar_sequence(source_sequence, target_sequences):
    max_match_length = -1
    most_similar_sequence = ""
    for target_sequence in target_sequences:
        match_length = SequenceMatcher(None, source_sequence, target_sequence).find_longest_match(0, len(source_sequence), 0, len(target_sequence)).size
        if max_match_length < match_length:
            max_match_length = match_length
            most_similar_sequence = target_sequence
    
    return most_similar_sequence

# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

# execute predicted sql with a time limitation
@func_set_timeout(120)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()
    

def decode_sqls(
    db_path,
    generator_outputs,
    batch_db_ids,
    batch_inputs,
    tokenizer,
    batch_tc_original
):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_sqls = []
    
    for batch_id in range(batch_size):
        pred_executable_sql = "sql placeholder"
        db_id = batch_db_ids[batch_id]
        db_file_path = db_path + "/{}/{}.sqlite".format(db_id, db_id)
        
        # print(batch_inputs[batch_id])
        # print("\n".join(tokenizer.batch_decode(generator_outputs[batch_id, :, :], skip_special_tokens = True)))

        for seq_id in range(num_return_sequences):
            cursor = get_cursor_from_path(db_file_path)
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens = True)

            pred_sql = pred_sequence.split("|")[-1].strip()
            pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
            
            try:
                # Note: execute_sql will be success for empty string
                assert len(pred_sql) > 0, "pred sql is empty!"

                results = execute_sql(cursor, pred_sql)
                # if the current sql has no execution error, we record and return it
                pred_executable_sql = pred_sql
                cursor.close()
                cursor.connection.close()
                break
            except Exception as e:
                print(pred_sql)
                print(e)
                cursor.close()
                cursor.connection.close()
            except FunctionTimedOut as fto:
                print(pred_sql)
                print(fto)
                del cursor
        
        final_sqls.append(pred_executable_sql)
    
    return final_sqls
