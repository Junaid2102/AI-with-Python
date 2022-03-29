graph = {
  'A': ['B', 'C'],
  'B': ['D', 'E'],
  'C': ['F', 'G'],
  'D': [],
  'E': [],
  'F': [],
  'G': []
}
print("==== Hello to advance Depth first Search Algorith ====")
visited = set()
way = set()
def dfs(graph, start, end, way, visited):
  way.append(start)
  visited.add(start)
  if start == end:
    return way
  for next in graph[start]:
    if next not in visited:
      answer = dfs(graph, next, end, way, visited)
      if answer is not None:
        return answer
    way.pop()
  return None

result = []
start = input("Enter starting node (btw A and F) : ")
end = input("Enter ending node (btw A and F) : ")
result = dfs(graph, start, end, result, visited)
print(result)