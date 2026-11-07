import «.formalsw».solvers.«13.py.ast»
import lang.py.FormalSW.HeapRuntimeCostComprehensions
import lang.py.FormalSW.IntegerListWork
import lang.py.FormalSW.HeapRuntimeInputs

namespace solvers.«13».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open HeapCalls.Runtime.NestedIntList (encode)
private abbrev program := Solvers.P13Ast.module

abbrev texts (items : List Int) : List HeapCalls.Value :=
  items.map (fun n => .scalar (.string (toString n)))

def formatting : Expr :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | .assign _ (.call _ (expression :: _)) :: _ => expression
  | _ => .constant .none

def formattingElement : Expr :=
  match formatting with | .listComp element _ => element | _ => .constant .none

private abbrev body (address : Nat) : List Stmt :=
  [.comprehensionAppend address formattingElement]

theorem formatting_element (heap : Heap) (env : HeapCalls.Env) (n : Int)
    (lookup : env "n" = some (.scalar (.int n))) :
    EvaluatesIn .word program env formattingElement heap (IntegerListWork.binarySize n+6)
      (.scalar (.string (toString n))) heap := by
  have run := EvaluatesIn.builtin_single (globals := []) (arithmetic := .word) (module := program)
    (builtin := .str) (heap := heap) (work := PrimitiveWork.bitLength n+1)
    (by intro h; cases h) (by intro h; cases h) (EvaluatesIn.lookup lookup) (by rfl) rfl
  exact run.mono (by have width := IntegerListWork.width_le_slot n; omega)

theorem formatting_loop (heap : Heap) (address : Nat) (items remaining : List Int)
    (found : heap[address]? = some (.list (encode items))) (position : Nat)
    (suffix : items.drop position = remaining) (env : HeapCalls.Env) (done : List Int) :
    ∃ finalEnv, BlockContinuesIn .word program env
      [.sequenceStepLoop "n" (.ref address) position (body heap.length) []]
      (heap ++ [.list (texts done)]) (18*IntegerListWork.size remaining+5)
      finalEnv (heap ++ [.list (texts (done ++ remaining))]) := by
  have inHeap : address < heap.length := List.getElem?_eq_some_iff.mp found |>.1
  have read : (heap ++ [.list (texts done)])[address]? = some (.list (encode items)) := by
    rw [List.getElem?_append_left inHeap]
    exact found
  have sequence : Mutable.sequence (.ref address) (heap ++ [.list (texts done)]) =
      some (.ok (encode items), heap ++ [.list (texts done)]) := by
    simp [Mutable.sequence, read]
  induction remaining generalizing position env done with
  | nil =>
      have absent : items[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      refine ⟨env, ?_⟩
      simp only [List.append_nil]
      intro following anchors stack
      apply advance
      · simp [step, run, sequenceValue, sequence, encode, absent]
      · simp [charge, read, PrimitiveWork.integer, IntegerListWork.size]
      · rfl
  | cons head tail ih =>
      let bound := setLocal env "n" (.scalar (.int head))
      have emitted := BlockContinuesIn.comprehension_append (module := program) (address := heap.length)
        (items := texts done) (formatting_element (heap ++ [.list (texts done)]) bound head rfl)
        (by simp)
      have written : BlockContinuesIn .word program bound (body heap.length)
          (heap ++ [.list (texts done)]) (IntegerListWork.binarySize head+13)
          bound (heap ++ [.list (texts (done ++ [head]))]) := by
        simpa [body, texts, List.map_append, Nat.add_assoc] using emitted
      obtain ⟨finalEnv, rest⟩ := ih (position+1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) bound (done ++ [head])
        (by simp [List.getElem?_append_left inHeap, found])
        (by simp [Mutable.sequence, List.getElem?_append_left inHeap, found])
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      refine ⟨finalEnv, ?_⟩
      intro following anchors stack
      have first : Reaches .word program anchors
          ⟨.block ([.sequenceStepLoop "n" (.ref address) position (body heap.length) []] ++ following) env,
            stack, heap ++ [.list (texts done)]⟩ 5
          ⟨.block (body heap.length ++ [.sequenceStepLoop "n" (.ref address) (position+1) (body heap.length) []] ++ following)
            bound, stack, heap ++ [.list (texts done)]⟩ :=
        advance (by simp [step, run, sequenceValue, sequence, encode, item, body, bound,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt])
          (by simp [charge, read, PrimitiveWork.integer]) rfl
      have combined := first.trans ((written.append rest) following anchors stack)
      simp only [List.append_assoc, List.singleton_append] at combined
      exact combined.mono (by rw [IntegerListWork.size_cons]; omega)

theorem formatting_cost (heap : Heap) (address : Nat) (items : List Int) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (encode items)))
    (lookup : env "nums" = some (.ref address)) :
    EvaluatesIn .word program env formatting heap (18*IntegerListWork.size items+13)
      (.ref heap.length) (heap ++ [.list (texts items)]) := by
  obtain ⟨finalEnv, loop⟩ := formatting_loop heap address items items found 0 rfl
    (comprehensionEnv env [("n", .name "nums", [])]) []
  have run := EvaluatesIn.list_comp_sequence (arithmetic := .word) (module := program)
    (element := formattingElement) (items := encode items) (size := items.length)
    (by intro expressions h; cases h) (EvaluatesIn.lookup lookup)
    (by simp [Mutable.sequence, found]) (by simp [sequenceLength, found, encode]) loop
  exact run.mono (by omega)

end solvers.«13».runtimeCost
