from sqlalchemy.orm import Query


def paginate(query: Query, page: int = 1, page_size: int = 10) -> dict:
  total = query.count()

  items = query.offset((page - 1) * page_size).limit(page_size).all()

  total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

  return {
      "items": items,
      "total": total,
      "page": page,
      "page_size": page_size,
      "total_pages": total_pages,
  }
