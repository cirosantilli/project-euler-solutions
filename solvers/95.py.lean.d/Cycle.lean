import solvers.«95.py.lean.d».Heap
import ProjectEulerSolutions.Lib.BoundedCycleReference
import ProjectEulerSolutions.Lib.IntegerListMinimumCalls

/-! Fresh cycle slices, their least member, and the source's longest-cycle comparison. -/
namespace solvers.«95».cycle
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open BoundedCycles BoundedCycleWalk BoundedCycleReference
set_option Elab.async false

abbrev program := Solvers.P95Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
variable {globals : Mutable.Globals}

theorem minimum (values : List Nat) (nonempty : values ≠ []) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (address : Nat)
    (found : heap[address]? = some (.list (values.map number)))
    (getCycle : env "cycle" = some (.ref address)) :
    Evaluates program env (.builtinCall .min [.name "cycle"]) heap (number (least values)) heap globals := by
  cases values with
  | nil => exact False.elim (nonempty rfl)
  | cons first rest =>
      have convert (values : List Nat) (initial : Nat) :
          (values.map (fun (n : Nat) => (n : Int))).foldl min (initial : Int) = ((values.foldl min initial : Nat) : Int) := by
        induction values generalizing initial with
        | nil => rfl
        | cons n values ih =>
            simp only [List.map_cons, List.foldl_cons]
            rw [show min (initial : Int) (n : Int) = (min initial n : Nat) by omega, ih]
      apply Evaluates.builtin (.cons (Evaluates.lookup getCycle) .nil)
      have returned := IntegerListMinimumCalls.minimum_primitive heap address (first : Int)
        (rest.map (fun (n : Nat) => (n : Int))) (by simpa only [TriangleBufferCalls.encode, List.map_cons,
          List.map_map, Function.comp_def, TriangleBufferCalls.number, number] using found)
      simpa only [least_fold, convert] using returned

abbrev sliceExpression : Expr := .subscript (.name "path")
  (.slice (some (.subscript (.name "idx") (.name "n"))) none none)

abbrev prepareCode : List Stmt :=
  [.assign [.name "cycle"] sliceExpression,
   .assign [.name "clen"] (.builtinCall .len [.name "cycle"]),
   .assign [.name "cmin"] (.builtinCall .min [.name "cycle"])]

def ready (env : HeapCalls.Env) (address : Nat) (values : List Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "cycle" (.ref address)) "clen" (number values.length))
    "cmin" (number (least values))

theorem ready_frame (env : HeapCalls.Env) (address : Nat) (values : List Nat) (name : String)
    (notCycle : name ≠ "cycle") (notLength : name ≠ "clen") (notMinimum : name ≠ "cmin") :
    ready env address values name = env name := by simp [ready, setLocal, notCycle, notLength, notMinimum]

theorem prepare (heap : Mutable.Heap) (env : HeapCalls.Env) (pathAddress positionsAddress current index : Nat)
    (path : List Nat) (positions : NaturalDictionary.Entries) (inside : index < path.length)
    (foundPath : heap[pathAddress]? = some (.list (path.map number)))
    (foundPositions : NaturalDictionary.Found heap positionsAddress positions)
    (getPath : env "path" = some (.ref pathAddress))
    (getPositions : env "idx" = some (.ref positionsAddress)) (getCurrent : env "n" = some (number current))
    (selected : NaturalDictionary.lookup positions current = some index) :
    BlockContinues program env prepareCode heap (ready env heap.length (path.drop index))
      (heap ++ [.list ((path.drop index).map number)]) globals := by
  let suffix := path.drop index
  let allocated := heap ++ [.list (suffix.map number)]
  let withCycle := setLocal env "cycle" (.ref heap.length)
  let withLength := setLocal withCycle "clen" (number suffix.length)
  have sliced : Evaluates program env sliceExpression heap (.ref heap.length) allocated globals := by
    apply Evaluates.slice (.cons (Evaluates.lookup getPath)
      (.cons (NaturalDictionary.lookup_evaluates heap env "idx" "n" positionsAddress current index positions
        foundPositions getPositions getCurrent selected) (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
    simpa only [← List.map_drop] using Mutable.getSlice_list_suffix heap pathAddress (path.map number) index foundPath
  have measured : Evaluates program withCycle (.builtinCall .len [.name "cycle"]) allocated
      (number suffix.length) allocated globals := by
    apply Evaluates.builtin (values := [.ref heap.length]) (.cons (Evaluates.lookup (by simp [withCycle, setLocal])) .nil)
    simp [primitive, Mutable.length, allocated, number]
  have leastValue := minimum (globals := globals) suffix (by
      intro empty
      have length := congrArg List.length empty
      simp only [suffix, List.length_drop, List.length_nil] at length
      omega) allocated withLength heap.length (by simp [allocated])
      (by simp [withLength, withCycle, setLocal])
  exact (BlockContinues.assign sliced).append
    ((BlockContinues.assign measured).append (BlockContinues.assign leastValue))

abbrev condition : Expr := .boolOp .or
  [.compare (.name "clen") [(.gt, .name "best_len")],
   .boolOp .and [.compare (.name "clen") [(.eq, .name "best_len")],
    .boolOp .or [.compare (.name "best_min") [(.eq, .constant (.int 0))],
      .compare (.name "cmin") [(.lt, .name "best_min")]]]]

abbrev replaceCode : List Stmt :=
  [.assign [.name "best_len"] (.name "clen"), .assign [.name "best_min"] (.name "cmin")]

def updated (env : HeapCalls.Env) (candidate : Nat × Nat) : HeapCalls.Env :=
  setLocal (setLocal env "best_len" (number candidate.1)) "best_min" (number candidate.2)

theorem test (best candidate : Nat × Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLength : env "clen" = some (number candidate.1))
    (getMinimum : env "cmin" = some (number candidate.2))
    (getBestLength : env "best_len" = some (number best.1))
    (getBestMinimum : env "best_min" = some (number best.2)) :
    Evaluates program env condition heap
      (.scalar (.bool (decide (best.1 < candidate.1 ∨
        candidate.1 = best.1 ∧ (best.2 = 0 ∨ candidate.2 < best.2))))) heap globals := by
  simp only [Bool.decide_or, Bool.decide_and]
  apply solvers.«95».heap.boolean_pair .or
  · exact Evaluates.compare (Evaluates.lookup getLength) (Evaluates.lookup getBestLength)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  apply solvers.«95».heap.boolean_pair .and
  · exact Evaluates.compare (Evaluates.lookup getLength) (Evaluates.lookup getBestLength)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  apply solvers.«95».heap.boolean_pair .or
  · exact Evaluates.compare (Evaluates.lookup getBestMinimum) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  · exact Evaluates.compare (Evaluates.lookup getMinimum) (Evaluates.lookup getBestMinimum)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem select_result (best candidate : Nat × Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLength : env "clen" = some (number candidate.1))
    (getMinimum : env "cmin" = some (number candidate.2))
    (getBestLength : env "best_len" = some (number best.1))
    (getBestMinimum : env "best_min" = some (number best.2)) :
    ∃ finalEnv, BlockContinues program env [.ifThen condition replaceCode []] heap finalEnv heap globals ∧
      finalEnv "best_len" = some (number (choose best candidate).1) ∧
      finalEnv "best_min" = some (number (choose best candidate).2) ∧
      ∀ name, name ≠ "best_len" → name ≠ "best_min" → finalEnv name = env name := by
  have tested := test (globals := globals) best candidate heap env getLength getMinimum getBestLength getBestMinimum
  by_cases better : best.1 < candidate.1 ∨ candidate.1 = best.1 ∧ (best.2 = 0 ∨ candidate.2 < best.2)
  · refine ⟨updated env candidate, ?_, ?_, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true) (by simpa only [better, decide_true] using tested) rfl
      exact (BlockContinues.assign (Evaluates.lookup getLength)).append
        (BlockContinues.assign (Evaluates.lookup (by simpa [setLocal] using getMinimum)))
    · simp [updated, setLocal, choose, better]
    · simp [updated, setLocal, choose, better]
    · intro name notLength notMinimum
      simp [updated, setLocal, notLength, notMinimum]
  · refine ⟨env, ?_, ?_, ?_, fun _ _ _ => rfl⟩
    · exact BlockContinues.branch (flag := false) (by simpa only [better, decide_false] using tested) rfl .nil
    · simpa only [choose, if_neg better] using getBestLength
    · simpa only [choose, if_neg better] using getBestMinimum

end solvers.«95».cycle
