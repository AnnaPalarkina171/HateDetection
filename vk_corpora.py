import vk
import vk_api
import pandas as pd
import numpy as np
from vk.exceptions import VkAPIError
from itertools import groupby


# Функция принимает список айди групп и создает словарь {group1_id: [post1_id, post2_id], group2_id: [...]} +
# получаем тексты постов
def get_tuple_id(groups_id):
    full_post_id = {}
    for group_id in groups_id:
        group_id = '-' + str(group_id)
        first = vk_api.wall.get(owner_id=group_id, v=5.92, count=100)  # Первое выполнение метода
        data = first["items"]
        count = first["count"] // 100
        # for i in range(1, count + 1):
        for i in range(1, 3):
            data = data + vk_api.wall.get(owner_id=group_id, v=5.92, count=100, offset=i * 100)["items"]
        posts_id = [post['id'] for post in data]
        full_post_id[group_id] = posts_id
        # post_text = [post['text'] for post in data if post['text'] != '']
        # with open('texts.txt', 'a', encoding='utf-8') as f:
        #     for text in post_text:
        #         f.write(text)
    return full_post_id


# Функция получает на вход словарь из {group1_id: [post1_id, post2_id], group2_id: [...]} и выдает тексты всех
# комментариев поста
def get_comments(full_post_id: dict):
    df = pd.read_csv('vk_corpora.csv')
    for group_id in full_post_id.keys():
        for post_id in full_post_id[group_id]:
            first = vk_api.wall.getComments(owner_id=group_id, post_id=post_id, v=5.92, count=100,
                                            preview_length=0)  # Первое выполнение метода
            data = first["items"]
            count = first["count"] // 100
            for i in range(1, count + 1):
                data = data + \
                       vk_api.wall.getComments(owner_id=group_id, v=5.92, count=100, offset=i * 100, preview_length=0,
                                               post_id=post_id)["items"]
            comments_text = [comment['text'] for comment in data if 'text' in comment.keys() if comment['text'] != '']

            for comment in comments_text:
                df = df.append({'Text': str(comment)}, ignore_index=True)
    df.to_csv('vk_corpora.csv', index=False)


if __name__ == "__main__":
    token = YOUR_TOKEN  # Сервисный ключ доступа
    session = vk.Session(access_token=token)  # Авторизация
    vk_api = vk.API(session, timeout=20)
    groups_id = ['29534144']
    get_comments(get_tuple_id(groups_id))

# для создания первоначального корпуса, чтобы потом только делать append
# data = pd.DataFrame({'Text':"Тёток с собой не брать-там на месте с избытком, можно выбрать."},index=[0])
# data.to_csv('vk_corpora.csv', index=False)
