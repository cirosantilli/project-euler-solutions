import Std
import ProjectEulerStatements.P79
import ProjectEulerSolutions.Termination.P79
namespace ProjectEulerSolutions.P79

def parseDigits (s : String) : List Nat :=
  s.data.filter (fun c => c >= '0' && c <= '9') |>.map (fun c => c.toNat - '0'.toNat)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def getAt (xs : List Nat) (i : Nat) : Nat :=
  match xs, i with
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => getAt xs i

termination_by 0
decreasing_by all_goals exact Termination.decreases
def addEdge (u v : Nat) (adj : Array (List Nat)) (indeg : Array Nat) : Array (List Nat) × Array Nat :=
  if adj[u]!.any (fun x => x == v) then
    (adj, indeg)
  else
    let adj := adj.set! u (v :: adj[u]!)
    let indeg := indeg.set! v (indeg[v]! + 1)
    (adj, indeg)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def buildGraph (attempts : List String) : Array (List Nat) × Array Nat × Array Bool :=
  let adj := Array.replicate 10 []
  let indeg := Array.replicate 10 0
  let present := Array.replicate 10 false
  let rec loop (xs : List String) (adj : Array (List Nat)) (indeg : Array Nat) (present : Array Bool)
      : Array (List Nat) × Array Nat × Array Bool :=
    match xs with
    | [] => (adj, indeg, present)
    | s :: xs =>
        let ds := parseDigits s
        let a := getAt ds 0
        let b := getAt ds 1
        let c := getAt ds 2
        let present := present.set! a true |>.set! b true |>.set! c true
        let (adj, indeg) := addEdge a b adj indeg
        let (adj, indeg) := addEdge b c adj indeg
        loop xs adj indeg present
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop attempts adj indeg present

termination_by 0
decreasing_by all_goals exact Termination.decreases
def topoSort (adj : Array (List Nat)) (indeg : Array Nat) (present : Array Bool) : List Nat :=
  let used := Array.replicate 10 false
  let total := (List.range 10).foldl (fun acc i => if present[i]! then acc + 1 else acc) 0
  let rec loop (k : Nat) (indeg : Array Nat) (used : Array Bool) (out : List Nat) : List Nat :=
    if k == total then
      out.reverse
    else
      let rec find (i : Nat) : Nat :=
        if i >= 10 then 10 else if present[i]! && used[i]! == false && indeg[i]! == 0 then i else find (i + 1)
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let v := find 0
      if v == 10 then
        out.reverse
      else
        let used := used.set! v true
        let rec decNeighbors (ns : List Nat) (indeg : Array Nat) : Array Nat :=
          match ns with
          | [] => indeg
          | x :: xs => decNeighbors xs (indeg.set! x (indeg[x]! - 1))
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        let indeg := decNeighbors (adj[v]!) indeg
        loop (k + 1) indeg used (v :: out)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 0 indeg used []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def derivePasscode (attempts : List String) : String :=
  let (adj, indeg, present) := buildGraph attempts
  let digits := topoSort adj indeg present
  digits.foldl (fun acc d => acc ++ toString d) ""

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (attempts : List String) : List Char :=
  (derivePasscode attempts).data

def serialize (chars : List Char) : String :=
  String.mk chars
end ProjectEulerSolutions.P79
