
// Combine frequently occuring consecutive NPs
//
// rule: look through the document at consecutive pairs of
// noun phrases.  For each pair determine if it is a candidate
// to combine. It is a candidate if all of the intermediate tokens are from
// a specified stop word list.
//
// For each candidate, keep a count of occurrences in the document.
// If a particular candidate occurs >= 2 times then combine the
// phrases.

#include <vector>
#include <string>
#include <unordered_map>
#include <set>

#ifndef UTILS
#include "_utils.cc"
#endif

// as we iterate through the document, need to keep n-gram counts
// so want a hash vector<string> -> something.
// accordingly need a custom hash for the vector<string>
struct VectorStringHasher
{
  std::size_t operator()(const std::vector<std::string>& tokens) const
  {
    // the hash function will lower case the tokens, join them
    // with a space and hash the string
    int nchars = 0;
    for (std::vector<std::string>::const_iterator it = tokens.begin();
        it != tokens.end(); ++it)
        nchars += (it->size() + 1);

    std::string hash_str;
    hash_str.reserve(nchars);
    for (std::vector<std::string>::const_iterator it = tokens.begin();
        it != tokens.end(); ++it)
        hash_str.append(*it);
        hash_str.push_back(' ');

    return std::hash<std::string>()(hash_str);
  }
};

typedef std::unordered_map<
    std::vector<std::string>,
    std::vector<std::pair<std::size_t, std::vector<std::size_t> > >,
    VectorStringHasher> counts_t;



class CombineState
{
    // state machine for combining NPs
    // we'll scan through the NPs in order and keep track of whether
    //  two NPs are candidate to combine, according to the rules
    //
    // States:
    // -1: nothing
    // 0: beginning of first NP
    // 1: between first and second
    // 2: in second NP

    public:
        CombineState();
        ~CombineState();

        // combine NPs in place
        void glob_np(std::vector<std::vector<iob_t> >& sentences);

    private:

        // counts: (token1, token2, token3) -> [
        //    [(sentenceid1, (indices of middle stop words)), ....]
        counts_t counts;
        int state;       // the internal state
        std::vector<std::string> candidate_tokens;
        std::vector<std::size_t> candidate_stopwords;

        std::set<std::string> stopwords;

        // reset the current candidate state
        void reset_candidate();

        void process_token(
            std::string& token, std::string& pos, char iob,
            std::size_t k, std::size_t i
        );

        void count_candidates(std::vector<std::vector<iob_t> >& sentences);

        // disable some default constructors
        CombineState& operator= (const CombineState& other);
        CombineState(const CombineState& other);

};

CombineState::CombineState() :
    counts(), state(-1), candidate_tokens(), candidate_stopwords(), stopwords()
{
    reset_candidate();

    stopwords.insert("the");
    stopwords.insert("a");
    stopwords.insert("in");
    stopwords.insert("of");
    stopwords.insert("an");
}

CombineState::~CombineState() {}

void CombineState::glob_np(std::vector<std::vector<iob_t> >& sentences)
{
    counts.clear();
    reset_candidate();

    count_candidates(sentences);

    // get the token strings we need to combine and combine them
    for (counts_t::iterator it = counts.begin(); it != counts.end(); ++it)
    {
        if (it->second.size() > 1)
        {
            // this token sequence occurs more then once, so combine
            // to combine set the stop word IOB labels to I
            // and the B at the beginning of the second NP to I
            for (std::size_t i = 0; i < it->second.size(); ++i)
            {
                std::size_t sentence_id = it->second[i].first;
                std::vector<std::size_t> stopword_indices = it->second[i].second;
                for (
                    std::vector<std::size_t>::iterator it_sw =
                        stopword_indices.begin();
                    it_sw != stopword_indices.end();
                    ++it_sw
                )
                    sentences[sentence_id][*it_sw].label = 'I';
                sentences[sentence_id][
                    stopword_indices[stopword_indices.size() - 1] + 1].label = 'I';
            }
        }
    }
}

void CombineState::reset_candidate()
{
    state = -1;
    candidate_tokens.clear();
    candidate_stopwords.clear();
}

void CombineState::count_candidates(std::vector<std::vector<iob_t> >& sentences)
{
    std::string np = "NP";
    for (std::size_t i = 0; i < sentences.size(); ++i)
    {
        reset_candidate();
        std::vector<iob_t> sentence = sentences[i];
        for (std::size_t k = 0; k < sentence.size(); ++k)
            process_token(
                sentence[k].token, sentence[k].tag, sentence[k].label, k, i);
        // if the sentence ends with an open candidate we need
        // to add it to the candidate list
        // this will force it to be added if necessary
        process_token(np, np, 'O', sentence.size(), i);
    }
}

void CombineState::process_token(
    std::string& token, std::string& pos, char iob, 
    std::size_t k, std::size_t i)
{
    if (state == -1)
    {
        if (iob == 'B')
        {
            state = 0;
            candidate_tokens.push_back(lower(token));
        }
        // else still in same state
    }
    else if (state == 0)
    {
        if (iob == 'B')
        {
            // beginning of a NP following a NP without stop words
            // not a candidate to combine
            reset_candidate();
            state = 0;
            candidate_tokens.push_back(lower(token));
        }
        else if (iob == 'O')
        {
            if (stopwords.find(lower(token)) != stopwords.end())
            {
                // an out of NP token in the stop word list
                state = 1;
                candidate_tokens.push_back(lower(token));
                candidate_stopwords.push_back(k);
            }
            else
            {
                // a out of NP token NOT in stop word list
                reset_candidate();
            }
        }
        else
        {
            // iob == 'I' still in same state
            candidate_tokens.push_back(lower(token));
        }
    }
    else if (state == 1)
    {
        if (iob == 'O')
        {
            if (stopwords.find(lower(token)) == stopwords.end())
            {
                // a token between stop words not in list, so we
                // don't potentially combine
                reset_candidate();
            }
            else
            {
                // token.lower() in self.stopwords
                // still in same state
                candidate_tokens.push_back(lower(token));
                candidate_stopwords.push_back(k);
            }
        }
        else if (iob == 'B')
        {
            state = 2;
            candidate_tokens.push_back(lower(token));
        }
    }
    else
    {
        // state == 2
        if (iob != 'I')
        {
            // end of second NP
            // add the potential candidate to the list
            counts[candidate_tokens].push_back(
                std::make_pair(i, candidate_stopwords));

            reset_candidate();

            if (iob == 'B')
            {
                state = 0;
                candidate_tokens.push_back(lower(token));
            }
            // else: iob == 'O', nothing to do
        }
        else
        {
            // iob == 'I', still in same state
            candidate_tokens.push_back(lower(token));
        }
    }
}



int main(void) { return 0; }



