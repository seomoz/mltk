
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

    def test_empty(self):
        '''
        The tagger works OK if it's passed empty tokens, or empty sentences
        '''
        sentences = [[], ['This', 'has', 'an', 'empty', '', 'token', '.']]
        tags = tagger.tag_sents(sentences)
        self.assertEqual(
            tags,
            [[], [('This', 'DT'), ('has', 'VBZ'), ('an', 'DT'),
            ('empty', 'JJ'), ('', 'NN'), ('token', 'NN'), ('.', '.')]]
        )

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

    def test_parenthesis(self):
        '''Parenthesis should get the right tags'''
        sentence = 'Here ( are [ some { parenthesis } ] in this sentence . )'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('Here', 'RB'),
             ('(', '('),
             ('are', 'VBP'),
             ('[', '('),
             ('some', 'DT'),
             ('{', '('),
             ('parenthesis', 'NN'),
             ('}', '}'),
             (']', ')'),
             ('in', 'IN'),
             ('this', 'DT'),
             ('sentence', 'NN'),
             ('.', '.'),
             (')', ')')])

        sentence = 'The USA ( United States of America ) is an acronym .'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('The', 'DT'),
             ('USA', 'NNP'),
             ('(', '('),
             ('United', 'NNP'),
             ('States', 'NNPS'),
             ('of', 'IN'),
             ('America', 'NNP'),
             (')', ')'),
             ('is', 'VBZ'),
             ('an', 'DT'),
             ('acronym', 'NN'),
             ('.', '.')])

    def test_negative_integers(self):
        sentence = '-1 plus 1 is equal to 0'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('-1', 'CD'),
             ('plus', 'CC'),
             ('1', 'CD'),
             ('is', 'VBZ'),
             ('equal', 'JJ'),
             ('to', 'TO'),
             ('0', 'CD')])

    def test_decimal_numbers(self):
        sentence = '0.5 is equal to 1 divided by 2'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('0.5', 'CD'),
             ('is', 'VBZ'),
             ('equal', 'JJ'),
             ('to', 'TO'),
             ('1', 'CD'),
             ('divided', 'VBN'),
             ('by', 'IN'),
             ('2', 'CD')])

    def test_decimal_no_leading_digit(self):
        sentence = '.5 is equal to 1 divided by 2'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('.5', 'CD'),
             ('is', 'VBZ'),
             ('equal', 'JJ'),
             ('to', 'TO'),
             ('1', 'CD'),
             ('divided', 'VBN'),
             ('by', 'IN'),
             ('2', 'CD')])

    def test_negative_decimal(self):
        sentence = '-0.5 is equal to -1 divided by 2'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('-0.5', 'CD'),
             ('is', 'VBZ'),
             ('equal', 'JJ'),
             ('to', 'TO'),
             ('-1', 'CD'),
             ('divided', 'VBN'),
             ('by', 'IN'),
             ('2', 'CD')])

        sentence = '-.5 is equal to -1 divided by 2'
        tags = tagger.tag(sentence.split())
        self.assertEqual(
            tags,
            [('-.5', 'CD'),
             ('is', 'VBZ'),
             ('equal', 'JJ'),
             ('to', 'TO'),
             ('-1', 'CD'),
             ('divided', 'VBN'),
             ('by', 'IN'),
             ('2', 'CD')])


if __name__ == '__main__':
    unittest.main()

