import ProjectEulerStatements.P83

namespace ProjectEulerSolutions.P83
open ProjectEulerStatements.P83 (bigVal getD2 neighbors)

def parseMatrix (text : String) : List (List Int) :=
  (text.splitOn "\n" |>.filter (fun line => line != "")).map fun line =>
    (line.splitOn ",").map fun value => (value.trimAscii.toString.toInt?).getD 0

structure SearchState where
  visited : Array Bool
  distances : Array Int

def findMin (total : Nat) (infinity : Int) (state : SearchState) : Nat × Int :=
  (List.range total).foldl (fun best index =>
    if !state.visited[index]! && state.distances[index]! < best.2
    then (index, state.distances[index]!) else best) (0, infinity)

def relax (matrix : List (List Int)) (width : Nat) (visited : Array Bool)
    (distance : Int) (distances : Array Int) (cell : Int × Int) : Array Int :=
  let index := cell.1.toNat * width + cell.2.toNat
  let candidate := distance + getD2 matrix cell.1 cell.2
  if !visited[index]! && candidate < distances[index]!
  then distances.set! index candidate else distances

def searchStep (matrix : List (List Int)) (state : SearchState) : SearchState :=
  let rows := matrix.length
  let cols := (matrix.headD []).length
  let infinity := bigVal matrix
  let (index, distance) := findMin (rows * cols) infinity state
  if distance = infinity then state
  else
    let visited := state.visited.set! index true
    let adjacent := neighbors (index / cols) (index % cols) rows cols
    let distances := adjacent.foldl (relax matrix cols visited distance) state.distances
    ⟨visited, distances⟩

def dijkstraMinPathSum (matrix : List (List Int)) : Int :=
  let total := matrix.length * (matrix.headD []).length
  if total = 0 then 0
  else
    let distances := (Array.replicate total (bigVal matrix)).set! 0 (getD2 matrix 0 0)
    let initial : SearchState := ⟨Array.replicate total false, distances⟩
    let final := (List.range total).foldl (fun state _ => searchStep matrix state) initial
    final.distances[total - 1]!

def solve (matrix : List (List Int)) : Int := dijkstraMinPathSum matrix

end ProjectEulerSolutions.P83
