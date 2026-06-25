#!/usr/bin/env python3
"""
全网牛奶盒图片爬取脚本
=======================
从多个搜索引擎爬取不同类型、品牌、规格的牛奶盒/牛奶包装图片。

核心解决: 图片源站防盗链 403 → 自定义 Session 带 Referer + 自定义 Downloader 重试

用法:
    python tools/crawl_milkbox_images.py                     # 全部预设类别
    python tools/crawl_milkbox_images.py --keywords "蒙牛纯牛奶盒,伊利纯牛奶盒" --max-num 100
    python tools/crawl_milkbox_images.py --keywords "苹果" --max-num 100
    python tools/crawl_milkbox_images.py --engine baidu --max-num 200
    python tools/crawl_milkbox_images.py --output ./data/milkbox_images
"""

import argparse
import random
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from PIL import Image

from icrawler.builtin import (
    BaiduImageCrawler,
    BingImageCrawler,
    GoogleImageCrawler,
)
from icrawler.downloader import ImageDownloader
from icrawler.utils.session import Session as ICrawlerSession

# ============================================================================
# 预设搜索词 — 覆盖不同品牌、类型、规格、场景
# ============================================================================
PRESET_QUERIES = {
    # ---- 真实场景：人物拿在手上 ----
    "真实场景_人物": [
        "woman drinking yogurt real life",
        "man holding milk carton",
        "child drinking milk bottle",
        "people drinking yogurt photo",
        "person holding milk box",
        "girl with yogurt glass",
        "breakfast milk real",
    ],
    # ---- 真实场景：喝/饮用 ----
    "真实场景_饮用": [
        "drinking yogurt from glass",
        "pouring milk from carton",
        "drinking milk real photo",
        "person drink yogurt",
        "milk beverage real life",
        "drinking almond milk",
        "oat milk drinking",
    ],
    # ---- 真实场景：日常/家庭 ----
    "真实场景_日常": [
        "milk on breakfast table",
        "yogurt daily life",
        "home milk snack",
        "family yogurt time",
        "kitchen milk box",
        "breakfast with milk photo",
        "yogurt with fruit real",
    ],
    # ---- 产品实拍：拿在手上 ----
    "产品实拍_手持": [
        "蒙牛牛奶 人物 拿 真实",
        "伊利酸奶 手拿 拍摄",
        "光明牛奶 人物喝 真实照片",
        "牛奶盒 人手拿",
        "酸奶 杯子 喝 人物",
        "yogurt hand hold real photo",
    ],
    # ---- 产品实拍：饮用场景 ----
    "产品实拍_饮用": [
        "蒙牛 喝 真实 照片",
        "伊利 喝 实拍",
        "牛奶 人喝 真实",
        "酸奶 饮用 真实场景",
        "milk drink photo real situation",
    ],
    # ---- 类型实拍：真实 ----
    "酸奶_真实": [
        "yogurt real photo person",
        "酸奶 真实 照片 人",
        "greek yogurt drinking real",
        "fruit yogurt eating real",
    ],
    "牛奶_真实": [
        "milk real photo drinking",
        "牛奶 真实 喝 照片",
        "fresh milk real life",
    ],
    # ---- 去掉高度广告性的词 ----
    "避免包装照": [
        "product shot milk not on white",
        "milk lifestyle photo",
        "casual milk moment",
    ],
}

# ============================================================================
# 全局计数器 — 跟踪跳过的失效链接
# ============================================================================
_skip_stats = {"404": 0, "403": 0, "429": 0, "503": 0, "400": 0, "500": 0, "network_err": 0}


def reset_skip_stats():
    for k in _skip_stats:
        _skip_stats[k] = 0


# ============================================================================
# 反防盗链: 多组 UA / Accept, 模拟不同浏览器
# ============================================================================
_USER_AGENTS = [
    # Chrome 124 / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 124 / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox 126 / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    # Edge 124 / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Safari / macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"
    " (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

_ACCEPT_HEADERS = [
    "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "image/webp,image/apng,image/*,*/*;q=0.8",
    "image/avif,image/webp,*/*",
    "*/*",
]


class RobustSession(ICrawlerSession):
    """
    自定义 Session (继承 icrawler Session, 兼容 proxy_pool):
    - 自动从图片 URL 派生 Referer
    - 每次请求随机 UA + Accept, 模拟真实浏览器
    - 添加 Connection: keep-alive 复用 TCP
    """

    def _pick_headers(self, url: str) -> dict:
        parsed = urlparse(url)
        # 用图片同域的主页作为 Referer (大部分防盗链只校验域名)
        referer = f"{parsed.scheme}://{parsed.netloc}/"

        return {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": random.choice(_ACCEPT_HEADERS),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": referer,
            "DNT": "1",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "Pragma": "no-cache",
        }

    def get(self, url: str, **kwargs):
        # 合并自定义头 (保留调用方通过 kwargs 传入的 headers 优先级)
        headers = self._pick_headers(url)
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        return super().get(url, headers=headers, **kwargs)


class RobustImageDownloader(ImageDownloader):
    """
    自定义图片下载器, 相比 icrawler 默认的 ImageDownloader:

    1. 用 RobustSession 替代普通 Session — 自带 Referer + 随机 UA
    2. 遇到 403 不再直接 break，而是换一组头重试
    3. 增加超时时间和重试次数
    4. 添加随机抖动避免被限速
    """

    def download(
        self,
        task,
        default_ext,
        timeout=15,         # 增加超时时间
        max_retry=3,        # 增加重试次数
        overwrite=False,
        **kwargs,
    ):
        file_url = task["file_url"]
        task["success"] = False
        task["filename"] = None
        retry = max_retry

        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext)
                if self.storage.exists(filename):
                    self.logger.info("skip downloading file %s", filename)
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get("reach_max_num"):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except Exception as e:
                _skip_stats["network_err"] += 1
                retry -= 1
                time.sleep(random.uniform(0.5, 1.5))  # 抖动
                if retry <= 0:
                    self.logger.warning(
                        "Network error on %s (exhausted retries): %s", file_url, e
                    )
                continue  # <-- 网络异常则重试
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break

                if response.status_code == 200:
                    if not self.keep_file(task, response, **kwargs):
                        _skip_stats["404"] += 1   # 图片损坏也归类跳过
                        break
                    with self.lock:
                        self.fetched_num += 1
                        filename = self.get_filename(task, default_ext)
                    self.logger.info("image #%s\t%s", self.fetched_num, file_url)
                    self.storage.write(filename, response.content)
                    task["success"] = True
                    task["filename"] = filename
                    break

                elif response.status_code in (403, 429, 503):
                    # 防盗链 / 限速 / 服务暂不可用 → 快速重试
                    retry -= 1
                    if retry > 0:
                        backoff = random.uniform(0.3, 1.0)
                        self.logger.info(
                            "status %d on %s, retry %.1fs (%d left)",
                            response.status_code, file_url, backoff, retry,
                        )
                        time.sleep(backoff)
                    else:
                        _skip_stats[str(response.status_code)] += 1
                        self.logger.info(
                            "status %d (exhausted retries), skipping: %s",
                            response.status_code, file_url,
                        )
                else:
                    # 404 / 410 等 → 链接已失效，跳过不重试
                    _skip_stats[str(response.status_code)] += 1
                    self.logger.info(
                        "status %d (dead link, skipped): %s",
                        response.status_code, file_url,
                    )
                    break
            finally:
                pass


# ============================================================================
# 辅助函数
# ============================================================================

def sanitize_filename(name: str, max_len: int = 60) -> str:
    """清理关键词，生成合法的目录名"""
    import re
    name = re.sub(r"[^\w一-鿿\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_len]


def count_images(directory: Path) -> int:
    """递归统计目录下的图片文件数"""
    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    return sum(
        1
        for f in directory.rglob("*")
        if f.suffix.lower() in exts
    )


def build_crawler(
    engine: str,
    save_dir: str,
):
    """
    创建带反防盗链能力的爬虫实例。

    核心注入点:
    - downloader_cls=RobustImageDownloader  (403 重试 + 随机头)
    - set_session → RobustSession           (Referer + 完整浏览器头)

    Args:
        engine: 搜索引擎名称 (bing / baidu / google)
        save_dir: 图片保存目录

    Returns:
        ImageCrawler 实例
    """
    crawler_cls = {
        "bing": BingImageCrawler,
        "baidu": BaiduImageCrawler,
        "google": GoogleImageCrawler,
    }
    cls = crawler_cls.get(engine.lower())
    if cls is None:
        raise ValueError(
            f"不支持的引擎 '{engine}', 可选: {list(crawler_cls.keys())}"
        )

    kwargs = {
        "downloader_cls": RobustImageDownloader,   # <-- 自定义下载器
        "downloader_threads": 4,
        "storage": {"root_dir": save_dir},
        "log_level": "WARNING",
    }

    if engine.lower() == "baidu":
        kwargs["downloader_threads"] = 3
    elif engine.lower() == "google":
        kwargs["parser_threads"] = 2

    crawler = cls(**kwargs)

    # --- 替换 Session 为 RobustSession ---
    robust_session = RobustSession(crawler.proxy_pool)
    # 继承爬虫初始化的 Accept-Language / UA（作为后备，实际每次 get 会覆盖）
    robust_session.headers.update(crawler.session.headers)
    crawler.session = robust_session
    # 下载器也指向新 session
    crawler.downloader.session = robust_session

    return crawler


def run_crawl(
    queries: dict,
    engine: str = "bing",
    output_root: str = "./milkbox_images",
    max_num_per_keyword: int = 50,
    filters: Optional[dict] = None,
):
    """
    执行批量爬取。

    Args:
        queries: {类别名: [关键词列表]} 字典
        engine: 主引擎
        output_root: 图片总输出目录
        max_num_per_keyword: 每个关键词最多下载张数
        filters: 图片过滤条件
    """
    output_root = Path(output_root)
    total_keywords = sum(len(kws) for kws in queries.values())

    reset_skip_stats()

    print(f"{'=' * 60}")
    print(f"图片爬取任务")
    print(f"{'=' * 60}")
    print(f"  类别数:      {len(queries)}")
    print(f"  总关键词数:  {total_keywords}")
    print(f"  每关键词:    最多 {max_num_per_keyword} 张")
    print(f"  预计总量:     ≤ {total_keywords * max_num_per_keyword} 张")
    print(f"  引擎:        {engine}")
    print(f"  输出目录:    {output_root.absolute()}")
    print(f"  防盗链处理:  Referer + 随机 UA + 403 重试")
    print(f"{'=' * 60}\n")

    # 展开所有任务: (category_name, keyword, save_subdir)
    tasks = []
    for category, keywords in queries.items():
        cat_dir = sanitize_filename(category)
        for kw in keywords:
            save_dir = str(output_root / cat_dir / sanitize_filename(kw))
            tasks.append((category, kw, save_dir))

    # 预创建目录
    for _, _, save_dir in tasks:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0
    total_images_before = count_images(output_root)
    t_start = time.time()

    for idx, (cat, kw, save_dir) in enumerate(tasks, 1):
        print(
            f"[{idx}/{len(tasks)}] 类别={cat} | 关键词=\"{kw}\""
        )

        try:
            crawler = build_crawler(engine=engine, save_dir=save_dir)
            crawler.crawl(
                keyword=kw,
                max_num=max_num_per_keyword,
                filters=filters or {},
                file_idx_offset="auto",
                max_idle_time=6,  # 无新图片 6s 后退出, 避免空等
            )
            completed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

        # 关键词间短暂休息
        if idx < len(tasks):
            time.sleep(0.5 + random.uniform(0, 0.5))

    elapsed = time.time() - t_start
    total_images = count_images(output_root) - total_images_before

    skipped_total = sum(_skip_stats.values())

    print(f"\n{'=' * 60}")
    print(f"爬取完成!")
    print(f"  成功关键词: {completed}")
    print(f"  失败关键词: {failed}")
    print(f"  实际下载:   {total_images} 张图片")
    if skipped_total > 0:
        print(f"  跳过/失效:  {skipped_total} 个链接 "
              f"(404: {_skip_stats['404']}, "
              f"403: {_skip_stats['403']}, "
              f"网络异常: {_skip_stats['network_err']})")
    print(f"  耗时:       {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  保存目录:   {output_root.absolute()}")
    print(f"{'=' * 60}")

    # 按类别统计
    if total_images:
        print(f"\n按类别统计:")
        for d in sorted(output_root.iterdir()):
            if d.is_dir():
                n = count_images(d)
                print(f"  {d.name}: {n} 张")
        print(f"\n  总计: {total_images} 张")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="全网牛奶盒图片爬取脚本 (反防盗链版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python crawl_milkbox_images.py
    python crawl_milkbox_images.py --categories 蒙牛,伊利 --max-num 100
    python crawl_milkbox_images.py --engine baidu --keywords "蒙牛纯牛奶,伊利安慕希"
    python crawl_milkbox_images.py --size large --color white
        """,
    )

    parser.add_argument(
        "--categories", type=str, default=None,
        help=f"预设类别 (逗号分隔), 可选: {list(PRESET_QUERIES.keys())}",
    )
    parser.add_argument(
        "--keywords", type=str, default=None,
        help="自定义搜索关键词 (逗号分隔), 覆盖 --categories",
    )
    parser.add_argument(
        "--engine", type=str, default="bing",
        choices=["bing", "baidu", "google"],
        help="搜索引擎 (默认: bing)",
    )
    parser.add_argument(
        "--max-num", type=int, default=50,
        help="每个关键词最多下载图片数 (默认: 50)",
    )
    parser.add_argument(
        "--output", type=str, default="./milkbox_images",
        help="图片输出根目录 (默认: ./milkbox_images)",
    )
    parser.add_argument(
        "--size", type=str, default=None,
        choices=["large", "medium", "icon", ">640x480", ">1024x768"],
        help="按图片尺寸过滤",
    )
    parser.add_argument(
        "--color", type=str, default=None,
        choices=["color", "blackandwhite", "transparent", "white",
                 "red", "orange", "yellow", "green", "blue", "purple",
                 "pink", "brown", "black", "gray", "teal"],
        help="按颜色过滤 (Bing 支持)",
    )
    parser.add_argument(
        "--type", type=str, default=None,
        choices=["photo", "clipart", "line", "animatedgif", "transparent"],
        help="按图片类型过滤",
    )

    args = parser.parse_args()

    # ---- 组装 queries ----
    if args.keywords:
        kw_list = [k.strip() for k in args.keywords.split(",") if k.strip()]
        queries = {"自定义": kw_list}
    elif args.categories:
        cat_list = [c.strip() for c in args.categories.split(",") if c.strip()]
        queries = {}
        for cat in cat_list:
            if cat in PRESET_QUERIES:
                queries[cat] = PRESET_QUERIES[cat]
            else:
                print(f"[WARN] 未知类别 '{cat}', 可选: {list(PRESET_QUERIES.keys())}")
    else:
        queries = PRESET_QUERIES.copy()

    # ---- 组装 filters ----
    filters = {}
    if args.size:
        filters["size"] = args.size
    if args.color:
        filters["color"] = args.color
    if args.type:
        filters["type"] = args.type

    # ---- 执行 ----
    run_crawl(
        queries=queries,
        engine=args.engine,
        output_root=args.output,
        max_num_per_keyword=args.max_num,
        filters=filters,
    )


if __name__ == "__main__":
    main()
