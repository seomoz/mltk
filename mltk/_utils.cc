
#include <string>
#include <algorithm>

#include "../ext/murmur3.c"


/**
    Define an interface for the taggers/chunkers/parsers/etc.

    These are stages of the pipeline that operate on a single sentence,
    as represented as a list of a templated types.  They have templated
    types for both input and output types.

    TIN is representation for a single word (e.g. string, (string, POS), etc
    TOUT is representation for the tagged version of word
    Subclasses implement tag_sentence(single sentence) and base class
        tags entire documents by looping over sentences
*/
template <class TIN, class TOUT>
class TaggerBase
{
    public:
        /// tags a single sentence.  Subclasses implement
        virtual std::vector<TOUT> tag_sentence(std::vector<TIN> const &) {}

        /// tag a document as a list of sentences
        void tag_sentences(std::vector<std::vector<TIN> >& document,
            std::vector<std::vector<TOUT> >& tags);

    private:

};

template <class TIN, class TOUT>
void TaggerBase<TIN, TOUT>::tag_sentences(
    std::vector<std::vector<TIN> >& document,
    std::vector<std::vector<TOUT> >& tags)
{
    tags.clear();
    tags.reserve(document.size());
    typename std::vector<std::vector<TIN> >::const_iterator it;
    for (it = document.begin(); it != document.end(); ++it)
        tags.push_back(tag_sentence(*it));
}


/// Use murmurhash as a custom hash for the string
#define SEED 5

uint64_t murmurhash3(const std::string& key)
{
    uint64_t ret[2];   // need 128 bits of space to hold result
    MurmurHash3_x64_128(key.c_str(), key.length(), SEED, ret);
    return ret[0];
}

std::string normalize(std::string const & word)
{
    /**< normalize a word.
     - All words are lower cased
     - 4 letter digits are represented as !YEAR
     - Other digits are represented as !DIGITS
    */
    if (word.find("-") != std::string::npos && word[0] != '-')
        return "!HYPHEN";
    else if (word.find_first_not_of("0123456789") == std::string::npos &&
        word.length() == 4)
        return "!YEAR";
    else if (std::isdigit(word[0]))
        return "!DIGITS";
    else
    {
        // lowercase
        std::string ret;
        ret.assign(word);
        std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
        return ret;
    }
}

// a variety of join functions, depending on how many strings to join...
const std::string SPACE = " ";

std::string join(std::string& s1, std::string& s2)
{
    /**< join with a space in the center
     find the length of the final string, reserve it then append */
    std::string sout;
    sout.reserve(s1.length() + s2.length() + 1);
    sout.append(s1);
    sout.append(SPACE);
    sout.append(s2);
    return sout;
}

std::string join(const char* s1, int n, std::string& s2)
{
    /**< join with a space in the center
     find the length of the final string, reserve it then append */
    std::string sout;
    sout.reserve(n + s2.length() + 1);
    sout.append(s1);
    sout.append(SPACE);
    sout.append(s2);
    return sout;
}

std::string join(const char* s1, int n, std::string& s2, std::string& s3)
{
    /**< join with a space in the center
     find the length of the final string, reserve it then append */
    std::string sout;
    sout.reserve(n + s2.length() + s3.length() + 2);
    sout.append(s1);
    sout.append(SPACE);
    sout.append(s2);
    sout.append(SPACE);
    sout.append(s3);
    return sout;
}

std::string join(const char* s1, int n, std::string& s2, std::string& s3,
    std::string& s4)
{
    /**< join with a space in the center
     find the length of the final string, reserve it then append */
    std::string sout;
    sout.reserve(n + s2.length() + s3.length() + s4.length() + 3);
    sout.append(s1);
    sout.append(SPACE);
    sout.append(s2);
    sout.append(SPACE);
    sout.append(s3);
    sout.append(SPACE);
    sout.append(s4);
    return sout;
}

std::string join(const char s1, const char& s2)
{
    /**< join with a space in the center
     find the length of the final string, reserve it then append */
    std::string sout;
    sout.reserve(3);
    sout.append(1, s1);
    sout.append(SPACE);
    sout.append(1, s2);
    return sout;
}


