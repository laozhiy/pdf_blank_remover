#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF空白页删除工具 - 优化版
自动扫描当前目录及子文件夹，自动删除空白页并替换原文件
优化版：提升性能、改善用户体验、增强稳定性
"""

import os
import sys
import fitz
import numpy as np
import fnmatch
import shutil
from typing import List, Tuple, Optional, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from pathlib import Path
import gc
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    file_path: str
    success: bool
    blank_pages_removed: int
    original_pages: int
    processing_time: float
    error_message: Optional[str] = None


class PDFBlankPageRemover:
    """PDF空白页删除工具 - 优化版"""
    
    def __init__(self, 
                 threshold: float = 0.005,
                 max_concurrent_files: int = 3,
                 enable_logging: bool = True,
                 log_level: LogLevel = LogLevel.INFO):
        """
        初始化优化版PDF空白页删除工具
        
        Args:
            threshold: 空白页检测阈值
            max_concurrent_files: 最大并发文件数
            enable_logging: 是否启用日志
            log_level: 日志级别
        """
        # 基本设置
        self.threshold = threshold
        self.create_backup = False
        self.output_suffix = "_无空白页"
        
        # 性能设置
        self.max_concurrent_files = max_concurrent_files
        self.auto_replace = True
        self.recursive_scan = True
        
        # 文件过滤设置
        self.include_patterns = ["*.pdf"]
        self.exclude_patterns = []
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_blank_pages': 0,
            'total_processing_time': 0.0,
            'errors': 0
        }
        
        # 日志设置
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging(log_level)
    
    def _setup_logging(self, log_level: LogLevel):
        """设置日志系统"""
        logging.basicConfig(
            level=getattr(logging, log_level.value),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pdf_processor.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _log(self, message: str, level: str = "INFO"):
        """统一日志记录"""
        if self.enable_logging and hasattr(self, 'logger'):
            getattr(self.logger, level.lower())(message)
        else:
            print(f"[{level}] {message}")
    
    def is_blank_page_optimized(self, page) -> bool:
        """
        优化版空白页检测算法
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            bool: 是否为空白页
        """
        try:
            # 1. 快速文本检测
            text = page.get_text().strip()
            if text:
                return False
            
            # 2. 获取页面图像
            pix = page.get_pixmap()
            if not pix or pix.width == 0 or pix.height == 0:
                return True
            
            # 3. 优化图像处理
            return self._analyze_page_image(pix)
            
        except Exception as e:
            self._log(f"检测页面时出错: {e}", "ERROR")
            return False
    
    def _analyze_page_image(self, pix) -> bool:
        """
        分析页面图像内容
        
        Args:
            pix: PyMuPDF pixmap对象
            
        Returns:
            bool: 是否为空白页
        """
        try:
            # 获取像素数据
            samples = pix.samples
            if not samples:
                return True
            
            # 计算图像尺寸
            width = pix.width
            height = pix.height
            total_pixels = width * height
            
            # 快速采样检测（提高性能）
            if total_pixels > 1000000:  # 大图像使用采样
                return self._sampled_analysis(samples, width, height)
            else:
                return self._full_analysis(samples, width, height)
                
        except Exception as e:
            self._log(f"图像分析时出错: {e}", "ERROR")
            return False
    
    def _sampled_analysis(self, samples: bytes, width: int, height: int) -> bool:
        """
        采样分析（适用于大图像）
        
        Args:
            samples: 像素数据
            width: 图像宽度
            height: 图像高度
            
        Returns:
            bool: 是否为空白页
        """
        try:
            # 采样比例
            sample_ratio = 0.1  # 采样10%的像素
            step = max(1, int(1 / sample_ratio))
            
            non_white_count = 0
            total_sampled = 0
            
            for y in range(0, height, step):
                for x in range(0, width, step):
                    if y * width * 3 + x * 3 + 2 < len(samples):
                        idx = y * width * 3 + x * 3
                        if idx + 2 < len(samples):
                            r, g, b = samples[idx], samples[idx+1], samples[idx+2]
                            if r < 240 or g < 240 or b < 240:
                                non_white_count += 1
                            total_sampled += 1
            
            if total_sampled == 0:
                return True
            
            non_white_ratio = non_white_count / total_sampled
            return non_white_ratio < self.threshold
            
        except Exception as e:
            self._log(f"采样分析时出错: {e}", "ERROR")
            return False
    
    def _full_analysis(self, samples: bytes, width: int, height: int) -> bool:
        """
        完整分析（适用于小图像）
        
        Args:
            samples: 像素数据
            width: 图像宽度
            height: 图像高度
            
        Returns:
            bool: 是否为空白页
        """
        try:
            # 转换为numpy数组
            img_array = np.frombuffer(samples, dtype=np.uint8)
            
            # 检查数据长度
            expected_size = width * height * 3
            if len(img_array) != expected_size:
                return self._fallback_analysis(samples)
            
            # 重塑为RGB图像
            img = img_array.reshape((height, width, 3))
            
            # 计算非白色像素比例
            non_white_pixels = np.sum(np.any(img < 240, axis=2))
            total_pixels = width * height
            non_white_ratio = non_white_pixels / total_pixels
            
            return non_white_ratio < self.threshold
            
        except Exception as e:
            self._log(f"完整分析时出错: {e}", "ERROR")
            return self._fallback_analysis(samples)
    
    def _fallback_analysis(self, samples: bytes) -> bool:
        """
        备用分析方法
        
        Args:
            samples: 像素数据
            
        Returns:
            bool: 是否为空白页
        """
        try:
            non_white_count = 0
            total_pixels = len(samples) // 3
            
            for i in range(0, len(samples), 3):
                if i + 2 < len(samples):
                    r, g, b = samples[i], samples[i+1], samples[i+2]
                    if r < 240 or g < 240 or b < 240:
                        non_white_count += 1
            
            non_white_ratio = non_white_count / total_pixels if total_pixels > 0 else 0
            return non_white_ratio < self.threshold
            
        except Exception as e:
            self._log(f"备用分析时出错: {e}", "ERROR")
            return False
    
    def _is_processed_file(self, file_path: str) -> bool:
        """
        检查文件是否已经被处理过
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否已处理
        """
        base_name = os.path.splitext(file_path)[0]
        return base_name.endswith(self.output_suffix)
    
    def scan_current_directory(self) -> List[str]:
        """
        扫描当前目录及子文件夹中的PDF文件
        
        Returns:
            List[str]: PDF文件路径列表
        """
        pdf_files = []
        current_dir = os.getcwd()
        
        self._log(f"扫描目录: {current_dir}")
        
        try:
            if self.recursive_scan:
                # 递归扫描
                for root, dirs, files in os.walk(current_dir):
                    # 跳过源码目录和其他不需要的目录
                    dirs[:] = [d for d in dirs if d not in ['源码', '__pycache__', '.git']]
                    
                    for file in files:
                        if any(fnmatch.fnmatch(file, pattern) for pattern in self.include_patterns):
                            file_path = os.path.join(root, file)
                            if not self._is_processed_file(file_path):
                                pdf_files.append(file_path)
            else:
                # 只扫描当前目录
                for file in os.listdir(current_dir):
                    if any(fnmatch.fnmatch(file, pattern) for pattern in self.include_patterns):
                        file_path = os.path.join(current_dir, file)
                        if not self._is_processed_file(file_path):
                            pdf_files.append(file_path)
            
            self._log(f"找到 {len(pdf_files)} 个PDF文件")
            return pdf_files
            
        except Exception as e:
            self._log(f"扫描目录时出错: {e}", "ERROR")
            return []
    
    def remove_blank_pages_replace(self, input_path: str) -> ProcessingResult:
        """
        删除空白页并直接替换原文件（优化版）
        
        Args:
            input_path: 输入PDF文件路径
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            # 打开PDF文件
            doc = fitz.open(input_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                self._log(f"文件 {input_path} 没有页面", "WARNING")
                doc.close()
                return ProcessingResult(
                    file_path=input_path,
                    success=False,
                    blank_pages_removed=0,
                    original_pages=0,
                    processing_time=time.time() - start_time,
                    error_message="文件没有页面"
                )
            
            # 找出非空白页
            non_blank_pages = []
            blank_count = 0
            
            for page_num in range(total_pages):
                page = doc[page_num]
                if not self.is_blank_page_optimized(page):
                    non_blank_pages.append(page_num)
                else:
                    blank_count += 1
                    self._log(f"  发现空白页: 第 {page_num + 1} 页")
            
            if blank_count == 0:
                self._log(f"文件 {input_path} 没有空白页")
                doc.close()
                return ProcessingResult(
                    file_path=input_path,
                    success=True,
                    blank_pages_removed=0,
                    original_pages=total_pages,
                    processing_time=time.time() - start_time
                )
            
            # 创建新文档
            new_doc = fitz.open()
            
            # 复制非空白页
            for page_num in non_blank_pages:
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # 保存到临时文件
            temp_path = input_path + ".tmp"
            new_doc.save(temp_path)
            new_doc.close()
            doc.close()
            
            # 替换原文件
            shutil.move(temp_path, input_path)
            
            processing_time = time.time() - start_time
            self._log(f"✓ 成功删除 {blank_count} 个空白页，文件已更新 (耗时: {processing_time:.2f}秒)")
            
            return ProcessingResult(
                file_path=input_path,
                success=True,
                blank_pages_removed=blank_count,
                original_pages=total_pages,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"处理文件 {input_path} 时出错: {e}"
            self._log(error_msg, "ERROR")
            return ProcessingResult(
                file_path=input_path,
                success=False,
                blank_pages_removed=0,
                original_pages=0,
                processing_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def process_current_directory(self) -> Tuple[int, int]:
        """
        处理当前目录中的所有PDF文件（优化版）
        
        Returns:
            Tuple[int, int]: (处理的文件数, 删除的空白页总数)
        """
        pdf_files = self.scan_current_directory()
        
        if not pdf_files:
            self._log("当前目录中没有找到PDF文件")
            return 0, 0
        
        self.stats['total_files'] = len(pdf_files)
        self._log(f"开始处理 {len(pdf_files)} 个PDF文件")
        
        processed_count = 0
        total_blank_pages = 0
        results = []
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.max_concurrent_files) as executor:
            future_to_file = {
                executor.submit(self.remove_blank_pages_replace, file_path): file_path 
                for file_path in pdf_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        processed_count += 1
                        total_blank_pages += result.blank_pages_removed
                        self.stats['total_processing_time'] += result.processing_time
                    else:
                        self.stats['errors'] += 1
                        
                except Exception as e:
                    self._log(f"处理文件 {file_path} 时出错: {e}", "ERROR")
                    self.stats['errors'] += 1
        
        # 更新统计信息
        self.stats['processed_files'] = processed_count
        self.stats['total_blank_pages'] = total_blank_pages
        
        # 显示处理结果
        self._display_processing_results(results)
        
        # 清理内存
        gc.collect()
        
        return processed_count, total_blank_pages
    
    def _display_processing_results(self, results: List[ProcessingResult]):
        """
        显示处理结果
        
        Args:
            results: 处理结果列表
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        self._log(f"\n{'='*50}")
        self._log(f"处理完成!")
        self._log(f"总文件数: {len(results)}")
        self._log(f"成功处理: {len(successful_results)}")
        self._log(f"处理失败: {len(failed_results)}")
        self._log(f"删除空白页总数: {sum(r.blank_pages_removed for r in successful_results)}")
        self._log(f"总处理时间: {sum(r.processing_time for r in results):.2f}秒")
        
        if failed_results:
            self._log(f"\n失败文件:")
            for result in failed_results:
                self._log(f"  - {result.file_path}: {result.error_message}")
        
        # 性能统计
        if successful_results:
            avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
            self._log(f"平均处理时间: {avg_time:.2f}秒/文件")
    
    def test_page(self, file_path: str, page_num: int):
        """
        测试指定页面是否为空白页
        
        Args:
            file_path: PDF文件路径
            page_num: 页面编号（从1开始）
        """
        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                self._log(f"页面编号 {page_num} 超出范围 (1-{len(doc)})", "ERROR")
                return
            
            page = doc[page_num - 1]
            is_blank = self.is_blank_page_optimized(page)
            
            self._log(f"文件: {file_path}")
            self._log(f"页面: {page_num}")
            self._log(f"是否为空白页: {'是' if is_blank else '否'}")
            
            if is_blank:
                self._log("检测结果: 空白页")
            else:
                self._log("检测结果: 非空白页")
            
            doc.close()
            
        except Exception as e:
            self._log(f"测试页面时出错: {e}", "ERROR")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.stats.copy()


def main():
    """主函数 - 优化版"""
    parser = argparse.ArgumentParser(description="PDF空白页删除工具 - 优化版")
    parser.add_argument('path', nargs='?', help='PDF文件或目录路径')
    parser.add_argument('--auto-scan', action='store_true', help='自动扫描当前目录及子文件夹')
    parser.add_argument('--threshold', type=float, default=0.005, help='设置检测阈值')
    parser.add_argument('--suffix', help='设置输出后缀')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份')
    parser.add_argument('--output-dir', help='设置输出目录')
    parser.add_argument('--test', help='测试模式: 文件路径,页码')
    parser.add_argument('--concurrent', type=int, default=3, help='并发处理数量')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    parser.add_argument('--no-logging', action='store_true', help='禁用日志')
    
    args = parser.parse_args()
    
    # 创建处理器
    remover = PDFBlankPageRemover(
        threshold=args.threshold,
        max_concurrent_files=args.concurrent,
        enable_logging=not args.no_logging,
        log_level=LogLevel(args.log_level)
    )
    
    # 应用命令行参数
    if args.suffix:
        remover.output_suffix = args.suffix
    if args.no_backup:
        remover.create_backup = False
    if args.output_dir:
        remover.custom_output_dir = args.output_dir
    
    # 测试模式
    if args.test:
        try:
            file_path, page_num = args.test.split(',')
            page_num = int(page_num.strip())
            remover.test_page(file_path.strip(), page_num)
        except ValueError:
            print("测试模式格式错误，请使用: --test '文件路径,页码'")
        return
    
    # 处理逻辑
    if args.auto_scan or (not args.path and not args.auto_scan):
        # 自动扫描当前目录及子文件夹
        print("🚀 自动扫描当前目录及子文件夹...")
        processed_count, total_blank_pages = remover.process_current_directory()
        
        if processed_count == 0:
            print("\n📖 使用方法:")
            print("  python pdf_blank_remover_optimized.py --auto-scan               # 自动扫描当前目录及子文件夹")
            print("  python pdf_blank_remover_optimized.py <PDF文件路径>             # 处理单个PDF文件")
            print("  python pdf_blank_remover_optimized.py <目录路径>                # 处理目录中的所有PDF")
            print("  python pdf_blank_remover_optimized.py --test <文件路径,页码>     # 测试单个页面")
            print("\n🔧 参数说明:")
            print("  --auto-scan    自动扫描当前目录及子文件夹（推荐）")
            print("  --threshold    设置检测阈值")
            print("  --suffix       设置输出后缀")
            print("  --no-backup    不创建备份")
            print("  --output-dir   设置输出目录")
            print("  --concurrent   设置并发处理数量")
            print("  --log-level    设置日志级别")
            print("  --test         测试模式")
        
        # 显示统计信息
        stats = remover.get_statistics()
        print(f"\n📊 处理统计:")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  成功处理: {stats['processed_files']}")
        print(f"  删除空白页: {stats['total_blank_pages']}")
        print(f"  处理时间: {stats['total_processing_time']:.2f}秒")
        print(f"  错误数量: {stats['errors']}")
        
        # 暂停，让用户能看到结果
        print("\n按任意键退出...")
        try:
            input()
        except KeyboardInterrupt:
            pass
        
    elif args.path:
        if os.path.isfile(args.path):
            # 处理单个文件
            print(f"📄 处理文件: {args.path}")
            result = remover.remove_blank_pages_replace(args.path)
            if result.success:
                print(f"✅ 处理完成，删除了 {result.blank_pages_removed} 个空白页")
            else:
                print(f"❌ 处理失败: {result.error_message}")
        elif os.path.isdir(args.path):
            # 处理目录
            print(f"📁 处理目录: {args.path}")
            os.chdir(args.path)
            processed_count, total_blank_pages = remover.process_current_directory()
            print(f"\n✅ 处理完成!")
            print(f"处理文件数: {processed_count}")
            print(f"删除空白页总数: {total_blank_pages}")
        else:
            print(f"❌ 路径不存在: {args.path}")


if __name__ == "__main__":
    main()
