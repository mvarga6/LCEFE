#include "mesh_reader.h"

using namespace std;

void ReadOptions::Include(ElementType type, int tag)
{
    // check if we have tags for this type
    bool exists = (TagsToRead.count(type) != 0);

    // if it doesn't exist in map
    if (!exists)
    {
        // instert empty vector under that type as key
        TagsToRead.insert(make_pair(type, vector<int>()));
    }

    // add the tag to that key's list
    TagsToRead[type].push_back(tag);
}