
import unittest

from mltk.aptagger import FastPerceptronTagger

tagger = FastPerceptronTagger()

class TestFastPerceptronTagger(unittest.TestCase):
    def test_tag(self):
        tags = tagger.tag(['The', 'first', 'sentence', '.'])
        self.assertEqual(
            tags,
            [('The', 'DT'), ('first', 'JJ'), ('sentence', 'NN'), ('.', '.')])

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

    def test_unicode(self):
        sentence = u'Beyonc\u00e9 performed during half time of Super Bowl'
        sentence += ' XLVII .'

        # unicode must be decoded to utf-8 or it raises an error
        self.assertRaises(UnicodeEncodeError, tagger.tag, sentence.split())

        # but things work if we encode
        tags = tagger.tag(sentence.encode('utf-8').split())
        self.assertEqual(
            tags,
            [('Beyonc\xc3\xa9', 'NNP'),
             ('performed', 'VBN'),
             ('during', 'IN'),
             ('half', 'DT'),
             ('time', 'NN'),
             ('of', 'IN'),
             ('Super', 'NNP'),
             ('Bowl', 'NNP'),
             ('XLVII', 'NNP'),
             ('.', '.')])


if __name__ == '__main__':
    unittest.main()

