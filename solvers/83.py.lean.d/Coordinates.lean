import solvers.«83.py.lean.d».NeighborSyntax

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

abbrev coordinatesAssignment : Stmt := .assign [.tuple [.name "r", .name "c"]]
  (.builtinCall .divmod [.name "u", .name "m"])

theorem coordinates_source : loopBody[4]? = some coordinatesAssignment := rfl

theorem coordinates_result (heap : Mutable.Heap) (env : HeapCalls.Env) (index width : Nat)
    (positive : 0 < width)
    (localIndex : env "u" = some (number index)) (localWidth : env "m" = some (number width)) :
    BlockContinues program env [coordinatesAssignment] heap
      (assignNames env ["r", "c"] [number (index / width : Nat), number (index % width : Nat)])
      (heap ++ [.tuple [number (index / width : Nat), number (index % width : Nat)]]) := by
  have evaluated : Evaluates program env (.builtinCall .divmod [.name "u", .name "m"])
      heap (.ref heap.length) (heap ++ [.tuple [number (index / width : Nat), number (index % width : Nat)]]) := by
    apply Evaluates.builtin (.cons (Evaluates.lookup localIndex) (.cons (Evaluates.lookup localWidth) .nil))
    have divided := Mutable.divmod_ints (index : Int) (width : Int) (by omega) heap
    have quotient : (index : Int).fdiv (width : Int) = ((index / width : Nat) : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
      rfl
    have remainder : (index : Int) - (width : Int) * (index : Int).fdiv (width : Int) = ((index % width : Nat) : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), ← Int.emod_def]
      rfl
    rw [remainder, quotient] at divided
    exact divided
  apply BlockContinues.unpack (names := ["r", "c"]) rfl evaluated
    (items := [number (index / width : Nat), number (index % width : Nat)]) ?_ rfl
  simp [Mutable.sequence]

end solvers.«83»
