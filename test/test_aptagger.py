
import unittest

from mltk.aptagger import FastPerceptronTagger

tagger = FastPerceptronTagger()

class TestFastPerceptronTagger(unittest.TestCase):
    def test_tag_sents(self):
        text_tags = [
            [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'),
            ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'),
            ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'),
            ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'),
            ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')],

            [('Mr.', 'NNP'), ('Vinken', 'NNP'), ('is', 'VBZ'),
            ('chairman', 'NN'), ('of', 'IN'), ('Elsevier', 'NNP'),
            ('N.V.', 'NNP'), (',', ','), ('the', 'DT'), ('Dutch', 'NNP'),
            ('publishing', 'NN'), ('group', 'NN'), ('.', '.')],

            [('Rudolph', 'NNP'), ('Agnew', 'NNP'), (',', ','),
            ('55', 'CD'), ('years', 'NNS'), ('old', 'JJ'), ('and', 'CC'),
            ('former', 'JJ'), ('chairman', 'NN'), ('of', 'IN'),
            ('Consolidated', 'NNP'), ('Gold', 'NNP'), ('Fields', 'NNP'),
            ('PLC', 'NNP'), (',', ','), ('was', 'VBD'), ('named', 'VBN'),
            ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'),
            ('of', 'IN'), ('this', 'DT'), ('British', 'JJ'),
            ('industrial', 'JJ'), ('conglomerate', 'NN'), ('.', '.')]]

        text = [[t[0] for t in sent] for sent in text_tags]
        tags = tagger.tag_sents(text)
        self.assertListEqual(tags, text_tags)

if __name__ == '__main__':
    unittest.main()

