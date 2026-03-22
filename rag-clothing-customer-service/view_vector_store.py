"""
向量库内容查看工具
用于查看 ChromaDB 向量库中存储的所有文档和元数据
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
import config_data as config


def view_vector_store():
    """查看向量库中的所有内容"""
    # 获取脚本所在目录的绝对路径
    script_dir = Path(__file__).parent.absolute()
    
    # 将相对路径转换为基于脚本目录的绝对路径
    if os.path.isabs(config.persist_directory):
        # 如果已经是绝对路径，直接使用
        persist_dir = config.persist_directory
    else:
        # 如果是相对路径，基于脚本目录转换为绝对路径
        persist_dir = str(script_dir / config.persist_directory)
    
    # 加载环境变量
    load_dotenv()
    
    # 获取 API key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("警告: 未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量")
        print("请先在 .env 或系统环境中配置 API key")
        return
    
    os.environ["DASHSCOPE_API_KEY"] = api_key
    
    try:
        print("=" * 80)
        print("向量库内容查看工具")
        print("=" * 80)
        print(f"脚本目录: {script_dir}")
        print(f"向量库路径: {persist_dir}")
        
        # 初始化 embedding 和向量库
        print("\n[初始化] 连接向量库...")
        embedding = DashScopeEmbeddings(model="text-embedding-v4")
        vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=embedding,
            persist_directory=persist_dir,
        )
        print("✓ 连接成功")
        
        # 获取所有文档
        print("\n[获取数据] 从向量库中获取所有文档...")
        all_docs = vector_store.get()
        
        total_count = len(all_docs.get('ids', []))
        print(f"✓ 共找到 {total_count} 个文档块")
        
        if total_count == 0:
            print("\n⚠️  向量库为空，请先上传文档")
            return
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("统计信息")
        print("=" * 80)
        print(f"文档块总数: {total_count}")
        print(f"Collection 名称: {config.collection_name}")
        print(f"存储路径: {persist_dir}")
        
        # 分析元数据
        metadatas = all_docs.get('metadatas', [])
        if metadatas:
            sources = {}
            for meta in metadatas:
                source = meta.get('source', '未知')
                sources[source] = sources.get(source, 0) + 1
            
            print(f"\n文档来源统计:")
            for source, count in sources.items():
                print(f"  - {source}: {count} 个文档块")
        
        # 显示详细内容
        print("\n" + "=" * 80)
        print("文档详细内容")
        print("=" * 80)
        
        ids = all_docs.get('ids', [])
        documents = all_docs.get('documents', [])
        
        for i, (doc_id, doc_content) in enumerate(zip(ids, documents), 1):
            print(f"\n[文档块 {i}/{total_count}]")
            print(f"ID: {doc_id}")
            
            # 显示元数据
            if metadatas and i <= len(metadatas):
                meta = metadatas[i-1]
                print(f"元数据:")
                for key, value in meta.items():
                    print(f"  - {key}: {value}")
            
            # 显示文档内容（限制长度）
            print(f"内容预览:")
            content_preview = doc_content[:200] if len(doc_content) > 200 else doc_content
            print(f"  {content_preview}")
            if len(doc_content) > 200:
                print(f"  ... (共 {len(doc_content)} 字符)")
            
            print("-" * 80)
        
        # 交互式查询测试
        print("\n" + "=" * 80)
        print("交互式查询测试")
        print("=" * 80)
        print("输入查询文本进行相似度搜索（输入 'exit' 退出）")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
        
        while True:
            try:
                query = input("\n请输入查询: ").strip()
                if query.lower() in ['exit', 'quit', '退出', 'q']:
                    break
                
                if not query:
                    continue
                
                print(f"\n正在搜索: '{query}'...")
                results = retriever.invoke(query)
                
                print(f"\n找到 {len(results)} 个相关文档:")
                for idx, doc in enumerate(results, 1):
                    print(f"\n[结果 {idx}]")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"元数据: {doc.metadata}")
                    content = doc.page_content[:300] if len(doc.page_content) > 300 else doc.page_content
                    print(f"内容: {content}")
                    if len(doc.page_content) > 300:
                        print(f"... (共 {len(doc.page_content)} 字符)")
                    print("-" * 80)
                    
            except KeyboardInterrupt:
                print("\n\n退出查询模式")
                break
            except Exception as e:
                print(f"查询出错: {str(e)}")
        
        print("\n" + "=" * 80)
        print("查看完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    view_vector_store()
