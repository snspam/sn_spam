import os
import util as ut


def purge():
    ut.out('purging...')

    domains = ['adclicks', 'ifwe', 'twitter', 'youtube', 'soundcloud',
               'russia', 'toxic', 'yelp_hotel', 'yelp_restaurant']

    folders_to_purge = ['independent/data/%s/folds/*',
                        'independent/output/%s/predictions/*',
                        'relational/output/%s/experiments/*',
                        'relational/output/%s/predictions/*',
                        'relational/mrf/*',
                        'relational/psl/data/%s/*']

    for domain in domains:
        for folder in folders_to_purge:
            path = folder % domain if '%s' in folder else folder
            os.system('rm -rf %s' % path)

if __name__ == '__main__':
    purge()
