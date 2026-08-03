#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
import requests

from PIL import Image
import pillow_heif


class BaiduOCR:

    def __init__(self, api_key, secret_key):

        """
        百度 OCR 初始化

        api_key:
            百度云 AK

        secret_key:
            百度云 SK
        """

        self.api_key = api_key
        self.secret_key = secret_key

        self.access_token = self.get_access_token()


        # 通用文字识别接口
        self.ocr_url = (
            "https://aip.baidubce.com/"
            "rest/2.0/ocr/v1/general_basic"
        )


    def get_access_token(self):

        """
        AK/SK 获取 access_token
        """

        url = (
            "https://aip.baidubce.com/oauth/2.0/token"
        )


        params = {

            "grant_type":
                "client_credentials",

            "client_id":
                self.api_key,

            "client_secret":
                self.secret_key
        }


        response = requests.post(
            url,
            params=params
        )


        result = response.json()


        if "access_token" not in result:

            raise RuntimeError(
                f"获取token失败: {result}"
            )


        print(
            "access_token 获取成功"
        )


        return result["access_token"]



    def convert_heic_to_jpg(
            self,
            image_path
    ):

        """
        HEIC 转 JPG

        百度OCR不支持HEIC
        """

        suffix = (
            os.path.splitext(image_path)[1]
            .lower()
        )


        if suffix not in [
            ".heic",
            ".heif"
        ]:

            return image_path



        print(
            "检测到HEIC，转换JPEG..."
        )


        heif = pillow_heif.read_heif(
            image_path
        )


        image = Image.frombytes(
            heif.mode,
            heif.size,
            heif.data
        )


        jpg_path = (
            os.path.splitext(image_path)[0]
            +
            ".jpg"
        )


        image.save(
            jpg_path,
            "JPEG",
            quality=95
        )


        print(
            f"转换完成: {jpg_path}"
        )


        return jpg_path




    def resize_image(
            self,
            image_path,
            max_size=1600
    ):

        """
        图片压缩

        防止上传过大
        """

        image = Image.open(
            image_path
        )


        w, h = image.size


        if max(w, h) <= max_size:

            return image_path



        scale = (
            max_size /
            max(w, h)
        )


        new_size = (
            int(w * scale),
            int(h * scale)
        )


        image = image.resize(
            new_size
        )


        output = (
            os.path.splitext(image_path)[0]
            +
            "_resize.jpg"
        )


        image.save(
            output,
            "JPEG",
            quality=90
        )


        print(
            f"resize完成: {output}"
        )


        return output




    def image_to_base64(
            self,
            image_path
    ):

        """
        图片转base64
        """

        with open(
            image_path,
            "rb"
        ) as f:

            data = f.read()


        return base64.b64encode(
            data
        ).decode(
            "utf-8"
        )




    def recognize(
            self,
            image_path
    ):

        """
        OCR识别入口
        """

        # 1. HEIC转换
        image_path = (
            self.convert_heic_to_jpg(
                image_path
            )
        )


        # 2. 图片resize
        image_path = (
            self.resize_image(
                image_path
            )
        )


        # 3. base64
        image_base64 = (
            self.image_to_base64(
                image_path
            )
        )


        url = (
            self.ocr_url
            +
            "?access_token="
            +
            self.access_token
        )


        headers = {

            "Content-Type":
            "application/x-www-form-urlencoded"
        }


        data = {

            "image":
            image_base64
        }


        response = requests.post(
            url,
            headers=headers,
            data=data
        )


        result = response.json()


        return result




if __name__ == "__main__":


    # =================================
    # 百度云 AK / SK
    # 后续替换
    # =================================

    AK = "your_AK"

    SK = "your_SK"



    ocr = BaiduOCR(
        api_key="RY133amnpx9bJywL8oA2ZVgv",
        secret_key="4NewCfq9GkaLJybVYJZq7WqZHxnnRp3B"
    )


    # 测试图片

    image_path = (
        "/root/taojianwei/projects/"
        "ultralytics/IMG_8640.HEIC"
    )


    result = ocr.recognize(
        image_path
    )


    print(
        "\n========== OCR RESULT =========="
    )


    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


    print(
        "\n========== TEXT =========="
    )


    if "words_result" in result:


        for item in result["words_result"]:

            print(
                item["words"]
            )

    else:

        print(
            "OCR失败"
        )