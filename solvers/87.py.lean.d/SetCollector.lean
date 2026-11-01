import solvers.«87.py.lean.d».Preparation
import lang.py.FormalSW.ScalarSetLemmas
import lang.py.FormalSW.LoopCalls
import ProjectEulerSolutions.Lib.ListEraseDups

namespace solvers.«87».search
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def keys (items : List Nat) : List Constant := items.eraseDups.map (fun (n : Nat) => .int n)
def working (base : Mutable.Heap) (items : List Nat) : Mutable.Heap := base ++ [.set (keys items)]

theorem insert_keys (items : List Nat) (value : Nat) :
    ScalarSet.insert (keys items) (.int value) = keys (items ++ [value]) := by
  have folded := ScalarSet.fold_insert_nats (items ++ [value])
  simpa only [List.map_append, List.map_cons, List.map_nil, List.foldl_append,
    List.foldl_cons, List.foldl_nil, ScalarSet.fold_insert_nats, keys] using folded

theorem found (base : Mutable.Heap) (items : List Nat) :
    (working base items)[base.length]? = some (.set (keys items)) := by simp [working]

theorem sequence (base : Mutable.Heap) (items values : List Nat) (address : Nat)
    (stored : base[address]? = some (.list (encode values))) :
    Mutable.sequence (.ref address) (working base items) = some (.ok (encode values), working base items) := by
  simp [working, Mutable.sequence, ProjectEulerSolutions.HeapCollection.prefix_found base _ address _ stored]

structure Addresses where
  squares : Nat
  cubes : Nat
  fourths : Nat

structure Locals (env : HeapCalls.Env) (limit : Nat) (addresses : Addresses) (collector : Nat) : Prop where
  bound : env "limit" = some (number limit)
  squares : env "squares" = some (.ref addresses.squares)
  cubes : env "cubes" = some (.ref addresses.cubes)
  fourths : env "fourths" = some (.ref addresses.fourths)
  set : env "representable" = some (.ref collector)

theorem Locals.set_other {env : HeapCalls.Env} {limit : Nat} {addresses : Addresses} {collector : Nat}
    (ready : Locals env limit addresses collector) (name : String) (value : Mutable.Value)
    (notLimit : name ≠ "limit") (notSquares : name ≠ "squares") (notCubes : name ≠ "cubes")
    (notFourths : name ≠ "fourths") (notSet : name ≠ "representable") :
    Locals (setLocal env name value) limit addresses collector := by
  rcases ready with ⟨hl, hs, hc, hf, hr⟩
  constructor <;> simp_all [setLocal, Ne.symm]

theorem add_value (base : Mutable.Heap) (items : List Nat) (value : Nat) (env : HeapCalls.Env)
    (getSet : env "representable" = some (.ref base.length)) (getValue : env "total" = some (number value)) :
    BlockContinues program env [.expr (.call (.attribute (.name "representable") "add") [.name "total"])]
      (working base items) env (working base (items ++ [value])) := by
  apply BlockContinues.discard
  have added := Evaluates.add (module := program) (globals := []) (Evaluates.lookup getSet) (found base items)
    (Evaluates.lookup getValue) (found base items)
  simpa [insert_keys, working, List.set_append_right] using added

theorem threshold (limit value : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (name : String)
    (getLimit : env "limit" = some (number limit)) (getValue : env name = some (number value)) :
    Evaluates program env (.compare (.name name) [(.gtE, .name "limit")]) heap
      (.scalar (.bool (decide (limit ≤ value)))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup getValue) (Evaluates.lookup getLimit)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

end solvers.«87».search
