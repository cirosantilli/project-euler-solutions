import ProjectEulerSolutions.Lib.HeapRangeFold
import ProjectEulerSolutions.Lib.TriangleBufferCalls
import lang.py.FormalSW.ZipCalls

/-! Integer pair comprehensions retain live source lists and actual zip tuples. -/
namespace ProjectEulerSolutions.IntegerZipCollection
open FormalSW.PyAst HeapCalls
open TriangleBufferCalls (number encode)
set_option Elab.async false

def working (base : Mutable.Heap) (output : List Int) (scratch : Mutable.Heap) : Mutable.Heap :=
  base ++ [.list (encode output)] ++ scratch

def target : Expr := .tuple [.name "a", .name "b"]
def body (element : Expr) (address : Nat) : List Stmt :=
  [.assign [target] (.name "$formalsw.unpack"), .comprehensionAppend address element]

theorem prefix_found (base scratch : Mutable.Heap) (output : List Int) (address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) : (working base output scratch)[address]? = some object := by
  unfold working
  rw [List.append_assoc, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem collect_loop (module : Module) (element : Expr) (combine : Int → Int → Int)
    (base : Mutable.Heap) (outer : HeapCalls.Env) (globals : Mutable.Globals)
    (evaluate : ∀ output garbage locals x y,
      HeapRangeFold.Frame ["a", "b", "$formalsw.unpack"] outer locals →
      locals "a" = some (number x) → locals "b" = some (number y) →
      Evaluates module locals element (working base output garbage) (number (combine x y))
        (working base output garbage) globals)
    (scratch : Mutable.Heap) (env : HeapCalls.Env) (left right output : List Int)
    (leftAddress rightAddress position : Nat)
    (getLeft : base[leftAddress]? = some (.list (encode left)))
    (getRight : base[rightAddress]? = some (.list (encode right))) (same : left.length = right.length)
    (safe : HeapRangeFold.Frame ["a", "b", "$formalsw.unpack"] outer env) :
    ∃ after finalScratch, BlockContinues module env
      [.zipStepLoop "$formalsw.unpack" [.ref leftAddress, .ref rightAddress] position (body element base.length) []]
      (working base output scratch) after
      (working base (output ++ List.zipWith combine (left.drop position) (right.drop position)) finalScratch) globals := by
  let heap := working base output scratch
  have read : zipItem [.ref leftAddress, .ref rightAddress] position heap =
      some (.ok (do let x ← left[position]?; let y ← right[position]?; pure [number x, number y]), heap) := by
    cases hx : left[position]? <;> cases hy : right[position]? <;>
      simp [zipItem, zipFields, sequenceValue, Mutable.sequence, prefix_found base scratch output _ _ getLeft,
        prefix_found base scratch output _ _ getRight, encode, hx, hy, Except.map, heap, number]
  by_cases active : position < left.length
  · have inRight : position < right.length := by omega
    let x := left[position]
    let y := right[position]
    let fields := [number x, number y]
    let more := scratch ++ [.tuple fields]
    let afterTuple := working base output more
    let tupleEnv := setLocal env "$formalsw.unpack" (.ref heap.length)
    let assigned := assignNames tupleEnv ["a", "b"] fields
    have fetched : zipItem [.ref leftAddress, .ref rightAddress] position heap = some (.ok (some fields), heap) := by
      simpa [List.getElem?_eq_getElem active, List.getElem?_eq_getElem inRight, fields, x, y] using read
    have allocated : Mutable.allocateTuple fields heap = some (.ok (.ref heap.length), afterTuple) := by
      simp [Mutable.allocateTuple, fields, heap, afterTuple, more, working, List.append_assoc, Except.map]
    have tupleFound : Mutable.sequence (.ref heap.length) afterTuple = some (.ok fields, afterTuple) := by
      have equal : afterTuple = heap ++ [.tuple fields] := by simp [afterTuple, heap, more, working, List.append_assoc]
      simp [equal, Mutable.sequence]
    have unpacked : BlockContinues module tupleEnv [.assign [target] (.name "$formalsw.unpack")] afterTuple assigned afterTuple globals :=
      BlockContinues.unpack rfl (Evaluates.lookup rfl) tupleFound rfl
    have assignedSafe : HeapRangeFold.Frame ["a", "b", "$formalsw.unpack"] outer assigned := by
      exact safe.trans ((HeapRangeFold.Frame.updated _ env "$formalsw.unpack" _ (by simp)).trans
        ((HeapRangeFold.Frame.updated _ tupleEnv "a" _ (by simp)).trans
          (HeapRangeFold.Frame.updated _ (setLocal tupleEnv "a" (number x)) "b" _ (by simp))))
    have evaluated : Evaluates module assigned element afterTuple (number (combine x y)) afterTuple globals :=
      evaluate output more assigned x y assignedSafe rfl rfl
    have emitted : BlockContinues module assigned [.comprehensionAppend base.length element] afterTuple assigned
        (working base (output ++ [combine x y]) more) globals := by
      have found : afterTuple[base.length]? = some (.list (encode output)) := by simp [afterTuple, working, List.append_assoc]
      have step := BlockContinues.comprehension_append evaluated found
      simpa [afterTuple, working, encode, List.append_assoc] using step
    obtain ⟨after, finalScratch, continued⟩ := collect_loop module element combine base outer globals evaluate more assigned left right (output ++ [combine x y])
      leftAddress rightAddress (position + 1) getLeft getRight same assignedSafe
    refine ⟨after, finalScratch, ?_⟩
    apply BlockContinues.zip_step fetched allocated
    simpa only [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
      List.drop_eq_getElem_cons active, List.drop_eq_getElem_cons inRight, List.zipWith_cons_cons,
      x, y, List.append_assoc, List.singleton_append, Bool.false_or, Bool.false_eq_true, if_false,
      List.cons_append, List.nil_append] using
      (unpacked.append emitted).append continued
  · have exhausted : zipItem [.ref leftAddress, .ref rightAddress] position heap = some (.ok none, heap) := by
      rw [read, List.getElem?_eq_none (by omega : left.length ≤ position)]
      rfl
    have gone : left.drop position = [] := List.drop_eq_nil_iff.mpr (by omega)
    refine ⟨env, scratch, ?_⟩
    simp only [gone, List.zipWith_nil_left, List.append_nil]
    exact BlockContinues.zip_done exhausted BlockContinues.nil
termination_by left.length - position
decreasing_by omega

/-- Capture both list references before allocating the result and its temporary pair tuples. -/
theorem collect (module : Module) (element : Expr) (combine : Int → Int → Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (leftAddress rightAddress : Nat) (left right : List Int)
    (getLeft : heap[leftAddress]? = some (.list (encode left)))
    (getRight : heap[rightAddress]? = some (.list (encode right))) (same : left.length = right.length)
    (expressions : List Expr)
    (arguments : Evaluates.Arguments module env expressions heap [.ref leftAddress, .ref rightAddress] heap globals)
    (evaluate : ∀ output garbage locals x y,
      HeapRangeFold.Frame ["a", "b", "$formalsw.unpack"] env locals →
      locals "a" = some (number x) → locals "b" = some (number y) →
      Evaluates module locals element (working heap output garbage) (number (combine x y))
        (working heap output garbage) globals) :
    ∃ extra, Evaluates module env (.listCompTargets element [(target, .zipIterable expressions, [])])
      heap (.ref heap.length) (working heap (List.zipWith combine left right) extra) globals := by
  have captured : zipSources [.ref leftAddress, .ref rightAddress] heap =
      some (.ok [.ref leftAddress, .ref rightAddress], heap) := by
    simp [zipSources, List.mapM, List.mapM.loop, Mutable.sequence, getLeft, getRight, sequenceSource, Except.map]
  let masked := comprehensionTargetsEnv env [(target, .zipIterable expressions, [])]
  have safe : HeapRangeFold.Frame ["a", "b", "$formalsw.unpack"] env masked := by
    intro name outside
    simp only [List.mem_cons, not_or] at outside
    simp [masked, comprehensionTargetsEnv, target, targetLocals, outside.1, outside.2.1]
  obtain ⟨after, scratch, loop⟩ := collect_loop module element combine heap env globals evaluate [] masked
    left right [] leftAddress rightAddress 0 getLeft getRight same safe
  refine ⟨scratch, ?_⟩
  apply Evaluates.list_comp_targets_zip arguments captured
  simpa [masked, working, body, target, comprehensionTargetName, comprehensionTargetBody,
    comprehensionFilters, comprehensionTargetsBody] using loop

end ProjectEulerSolutions.IntegerZipCollection
