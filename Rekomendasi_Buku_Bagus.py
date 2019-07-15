import pandas as pd
import numpy as np

dfb = pd.read_csv('books.csv')
dfr = pd.read_csv('ratings.csv')

# drop data kolom
dfb = dfb.drop(
    [
        'goodreads_book_id', 'best_book_id', 'work_id', 'books_count',
        'isbn', 'isbn13', 'original_publication_year', 'work_ratings_count',
        'work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
        'image_url', 'small_image_url', 'ratings_count'
    ],
    axis=1
)

# data join
def mergeCol(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])
dfb['Features']=dfb.apply(mergeCol,axis='columns')
# print(dfb.head())

# vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model=CountVectorizer(tokenizer=lambda x:x.split(' '))
matrixFeature=model.fit_transform(dfb['Features'])

features=model.get_feature_names()
jmlFeatures=len(features)

# cosine
from sklearn.metrics.pairwise import cosine_similarity
score=cosine_similarity(matrixFeature)

# input user
andi1=dfb[dfb['original_title']=='The Hunger Games']['book_id'].index.values[0]
andi2=dfb[dfb['original_title']=='Catching Fire']['book_id'].index.values[0] 
andi3=dfb[dfb['original_title']=='Mockingjay']['book_id'].index.values[0]
andi4=dfb[dfb['original_title']=='The Hobbit or There and Back Again']['book_id'].index.values[0] 
suka1=[andi1,andi2,andi3,andi4]

budi1=dfb[dfb['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].index.values[0] 
budi2=dfb[dfb['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].index.values[0]
budi3=dfb[dfb['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].index.values[0]
suka2=[budi1,budi2,budi3]

ciko1=dfb[dfb['original_title']=='Robots and Empire']['book_id'].index.values[0]
suka3=[ciko1]

dedi1=dfb[dfb['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].index.values[0]
dedi2=dfb[dfb['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].index.values[0] 
dedi3=dfb[dfb['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].index.values[0]
suka4=[dedi1,dedi2,dedi3]

ello1=dfb[dfb['original_title']=='Doctor Sleep']['book_id'].index.values[0]
ello2=dfb[dfb['original_title']=='The Story of Doctor Dolittle']['book_id'].index.values[0] 
ello3=dfb[dfb['title']=='Bridget Jones\'s Diary (Bridget Jones, #1)']['book_id'].index.values[0] 
suka5=[ello1,ello2,ello3]

daftarScoreAndi1=list(enumerate(score[andi1]))
daftarScoreAndi2=list(enumerate(score[andi2]))
daftarScoreAndi3=list(enumerate(score[andi3]))
daftarScoreAndi4=list(enumerate(score[andi4]))

daftarScoreBudi1=list(enumerate(score[budi1]))
daftarScoreBudi2=list(enumerate(score[budi2]))
daftarScoreBudi3=list(enumerate(score[budi3]))

daftarScoreCiko=list(enumerate(score[ciko1]))

daftarScoreDedi1=list(enumerate(score[dedi1]))
daftarScoreDedi2=list(enumerate(score[dedi2]))
daftarScoreDedi3=list(enumerate(score[dedi3]))

daftarScoreEllo1=list(enumerate(score[ello1]))
daftarScoreEllo2=list(enumerate(score[ello2]))
daftarScoreEllo3=list(enumerate(score[ello3]))


daftarScoreAndi=[]
for i in daftarScoreAndi1:
    daftarScoreAndi.append((i[0],0.25*(daftarScoreAndi1[i[0]][1]+daftarScoreAndi2[i[0]][1]+daftarScoreAndi3[i[0]][1]+daftarScoreAndi4[i[0]][1])))
daftarScoreBudi=[]
for i in daftarScoreAndi1:
    daftarScoreBudi.append((i[0],(daftarScoreBudi1[i[0]][1]+daftarScoreBudi2[i[0]][1]+daftarScoreBudi3[i[0]][1])/3))
daftarScoreDedi=[]
for i in daftarScoreAndi1:
    daftarScoreDedi.append((i[0],(daftarScoreDedi1[i[0]][1]+daftarScoreDedi2[i[0]][1]+daftarScoreDedi3[i[0]][1])/3))
daftarScoreEllo=[]
for i in daftarScoreAndi1:
    daftarScoreEllo.append((i[0],(daftarScoreEllo1[i[0]][1]+daftarScoreEllo2[i[0]][1]+daftarScoreEllo3[i[0]][1])/3))

# sort
sortDaftarScoreandi=sorted(
    daftarScoreAndi,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScorebudi=sorted(
    daftarScoreBudi,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreCiko=sorted(
    daftarScoreCiko,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreDedi=sorted(
    daftarScoreDedi,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreEllo=sorted(
    daftarScoreEllo,
    key=lambda j:j[1],
    reverse=True
)

# similarity
similarBooksandi=[]
for i in sortDaftarScoreandi:
    if i[1]>0:
        similarBooksandi.append(i)
similarBooksbudi=[]
for i in sortDaftarScorebudi:
    if i[1]>0:
        similarBooksbudi.append(i)
similarBooksCiko=[]
for i in sortDaftarScoreCiko:
    if i[1]>0:
        similarBooksCiko.append(i)
similarBooksDedi=[]
for i in sortDaftarScoreDedi:
    if i[1]>0:
        similarBooksDedi.append(i)
similarBooksEllo=[]
for i in sortDaftarScoreEllo:
    if i[1]>0:
        similarBooksEllo.append(i)

print('1. Buku bagus untuk andi:')
for i in range(0,5):
    if similarBooksandi[i][0] not in suka1:
        print('-',dfb['original_title'].iloc[similarBooksandi[i][0]])
    else:
        i+=5
        print('-',dfb['original_title'].iloc[similarBooksandi[i][0]])

print(' ')
print('2. Buku bagus untuk budi:')
for i in range(0,5):
    if similarBooksbudi[i][0] not in suka2:
        print('-',dfb['original_title'].iloc[similarBooksbudi[i][0]])
    else:
        i+=5
        print('-',dfb['original_title'].iloc[similarBooksbudi[i][0]])

print(' ')
print('3. Buku bagus untuk ciko:')
for i in range(0,5):
    if similarBooksCiko[i][0] not in suka3:
        print('-',dfb['original_title'].iloc[similarBooksCiko[i][0]])
    else:
        i+=5
        print('-',dfb['original_title'].iloc[similarBooksCiko[i][0]])

print(' ')
print('4. Buku bagus untuk Dedi:')
for i in range(0,5):
    if similarBooksDedi[i][0] not in suka4:
        print('-',dfb['original_title'].iloc[similarBooksDedi[i][0]])
    else:
        i+=5
        print('-',dfb['original_title'].iloc[similarBooksDedi[i][0]])

print(' ')
print('5. Buku bagus untuk Ello:')
for i in range(0,5):
    if similarBooksEllo[i][0] not in suka5:
        if str(dfb['original_title'].iloc[similarBooksEllo[i][0]])=='nan':
            print('-',dfb['title'].iloc[similarBooksEllo[i][0]])
        else:
            print('-',dfb['original_title'].iloc[similarBooksEllo[i][0]])  
    else:
        i+=5
        if str(dfb['original_title'].iloc[similarBooksEllo[i][0]])=='nan':
            print('-',dfb['title'].iloc[similarBooksEllo[i][0]])
        else:
            print('-',dfb['original_title'].iloc[similarBooksEllo[i][0]]) 