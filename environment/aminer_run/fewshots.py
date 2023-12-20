"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

REACT_EXAMPLE = """Question: What are the research interests of Jianguang Lou from Microsoft Research Asia?
Thought 1: The question asks for a person's information. I only need to search Jianguang Lou and find his research interests.
Action 1: searchPerson(name='Jianguang Lou', organization='Microsoft Research Asia')
Observation 1: One person found: {"contact": {"address": "", "affiliation": "Microsoft Research Asia", "affiliationZh": "", "bio": "I mainly worked on conversational bot technologies especially designed for data analysis domain, which can help common users to analyze tabular data sets through natural language conversation.", "bioZh": "", "edu": "Automation institute, CAS<br>PhD pattern recognition<br>2000 - 2003<br><br>Zhejiang University<br>batchler & graduate Automation<br>1993 - 2000", "eduZh": "", "email": "jlou@microsoft.com", "fax": "", "homepage": "https://www.microsoft.com/en-us/research/people/jlou/", "phone": "", "position": "Other", "positionZh": "\u5176\u4ed6", "work": "Microsoft Research Asia<br>Principal Researcher<br>2005.7 - \u81f3\u4eca<br>Postdoctoral Researcher<br>2003. 8 - 2005.7", "workZh": ""}, "gender": "male", "gindex": 78, "hindex": 36, "id": "53f431b4dabfaedce54fc81a", "language": "chinese", "name": "Jianguang Lou", "nation": "China", "ncitation": 6390, "npubs": 154, "orgZh": "", "interests": ["Log Analysis", "Data Mining", "Online Service System", "Image Sequences", "Software Analytics", "Epipolar Geometry", "Image Motion Analysis", "Video Coding", "Incident Management", "Problem Identification"]}
Thought 2: Jianguang Lou's research interests was included in the "interests" key. So I just need to return the interests.
Action 2: Finish(["Log Analysis", "Data Mining", "Online Service System", "Image Sequences", "Software Analytics", "Epipolar Geometry", "Image Motion Analysis", "Video Coding", "Incident Management", "Problem Identification"])

Question: Who are the collaborators of Jinxiong Xia in the field of Dialogue Topic Segmentation?
Thought 1: I need to search Jinxiong Xia, find his id, then get his collaborators.
Action 1: searchPerson(name='Jinxiong Xia', interest='Dialogue Topic Segmentation')
Observation 1: One person found: {"contact": {"address": "", "affiliation": "Peking University", "affiliationZh": "", "bio": "", "bioZh": "", "edu": "", "eduZh": "", "email": "xiajx98@gmail.com", "fax": "", "homepage": "https://www.youtube.com/watch?v=Ms6ri_x9yLM", "phone": "", "position": "", "positionZh": "", "work": "", "workZh": ""}, "gender": "unknown", "gindex": 1, "hindex": 1, "id": "64c32e168eff4696a89772b6", "language": "chinese", "name": "Jinxiong Xia", "ncitation": 3, "npubs": 2, "interests": ["Dialogue Topic Segmentation", "Parallel Extraction", "Boundary Ambiguity", "Data Noise", "Neighbor Smoothing", "EDTA", "Elisa", "EMSA", "EP", "FJC"]}
Thought 2: Jinxiong Xia's id is 64c32e168eff4696a89772b6. I need to use the getCoauthors API to find his collaborators.
Action 2: getCoauthors('64c32e168eff4696a89772b6')
Observation 2: [{'person_id': '562b857945cedb3398a8c714', 'name': 'Fan Yang', 'relation': 'coauthor'}]
Thought 3: Fan Yang is the collaborator of Jinxiong Xia. I just need to return Fan Yang.
Action 3: Finish(["Fan Yang"])
"""
