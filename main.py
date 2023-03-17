import openreview
import numpy as np
import pandas as pd
from tqdm import tqdm

client = openreview.Client(baseurl='https://api.openreview.net')
for year in [2021, 2022, 2023]:

    submissions = client.get_all_notes(invitation=f"ICLR.cc/{year}/Conference/-/Blind_Submission", details='directReplies')

    reviews = []
    for submission in submissions:
        reviews = reviews + [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Official_Review")]

    scores_dict = {}
    for r in reviews:
        if r['forum'] not in scores_dict:
            scores_dict[r['forum']] = []
        try:
            scores_dict[r['forum']].append(int(r['content']['recommendation'].split(':')[0]))
        except:
            scores_dict[r['forum']].append(int(r['content']['rating'].split(':')[0]))

    statistics = []
    all_data, abstracts = [], []
    for s in tqdm(submissions):
        abstract = s.content['abstract']
        keywords = s.content['keywords']
        if s.forum not in scores_dict:
            continue
        title = s.content['title']
        if 'Please_choose_the_closest_area_that_your_submission_falls_into' in s.content:
            area = s.content['Please_choose_the_closest_area_that_your_submission_falls_into']
        else:
            area = ''
        scores = scores_dict[s.forum]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        _title = title.lower()
        _keywords = [k.lower() for k in keywords]
        _abstract = abstract.lower()
        if (
            (('graph ' in _keywords) or ('graph ' in _abstract) or ('graph ' in _title) or \
            (' graph' in _keywords) or (' graph' in _abstract) or (' graph' in _title)) and \
            (not 'graphics' in _keywords) and (not 'graphics' in _abstract) and (not 'graphics' in _title)
        ) or \
        ('gnn' in _keywords) or ('gnn' in _abstract) or ('gnn' in _title) or \
        ('gcn' in _keywords) or ('gcn' in _abstract) or ('gcn' in _title):
                
            all_data.append([ title, str(avg_score), str(std_score), ';'.join([str(i) for i in scores]), area, keywords, abstract])
            abstracts.append(abstract)


    df = pd.DataFrame(all_data, columns=['Title', 'Average Score', 'Standard Deviation', 'Individual Scores', 'Author-defined Area', 'Keywords', 'Abstract'])
    df = df.sort_values(by=['Average Score'], ascending=False, ignore_index=True)
    df.index = np.arange(1, len(df)+1)
    df.to_csv('output.csv', index='True')
    
    # writing to file
    txt = ''
    for abstract in abstracts:
        txt += abstract + '\n'
    with open(f'abstracts/{year}.txt', 'w') as f:
        f.write(txt)