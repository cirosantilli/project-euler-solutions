import lang.py.FormalSW.HeapMemoCallRules

/-! Normal block completion with heap-dependent tuple-cache allocation. -/
namespace ProjectEulerSolutions.HeapMemoBlock
open FormalSW.PyAst HeapCalls HeapMemoCalls
set_option Elab.async false
variable {module : Module} {spec : Specification} {globals : Mutable.Globals}

/-- A block reaches the following statements with its actual locals and heap. -/
def Continues (module : Module) (spec : Specification) (env : HeapCalls.Env) (body : List Stmt)
    (heap : Mutable.Heap) (post : HeapCalls.Env → Mutable.Heap → Prop) (globals : Mutable.Globals := []) : Prop :=
  ∀ following anchors stack, HeapMemoCalls.RunsRel module spec anchors
    ⟨.block (body ++ following) env, stack, heap⟩
    (fun final => ∃ afterEnv after, final = ⟨.block following afterEnv, stack, after⟩ ∧ post afterEnv after) globals

namespace Continues

variable {env : HeapCalls.Env} {body rest : List Stmt} {heap : Mutable.Heap}
variable {pre post : HeapCalls.Env → Mutable.Heap → Prop}

theorem nil (holds : post env heap) : Continues module spec env [] heap post globals :=
  fun _ _ _ => RunsRel.refl ⟨env, heap, rfl, holds⟩

theorem consequence (run : Continues module spec env body heap pre globals)
    (implies : ∀ afterEnv after, pre afterEnv after → post afterEnv after) :
    Continues module spec env body heap post globals := by
  intro following anchors stack
  apply (run following anchors stack).consequence
  rintro final ⟨afterEnv, after, rfl, holds⟩
  exact ⟨afterEnv, after, rfl, implies afterEnv after holds⟩

theorem from_heap (disjoint : CacheDisjoint module module)
    (run : HeapCalls.BlockContinues module env body heap afterEnv after globals)
    (preserved : Cache.Preserves spec heap after) (holds : post afterEnv after) :
    Continues module spec env body heap post globals := by
  intro following anchors stack
  apply (Runs.from_heap (Extends.refl module) disjoint (run following anchors stack) preserved).consequence
  rintro final rfl
  exact ⟨afterEnv, after, rfl, holds⟩

theorem append (first : Continues module spec env body heap pre globals)
    (second : ∀ afterEnv after, pre afterEnv after → Continues module spec afterEnv rest after post globals) :
    Continues module spec env (body ++ rest) heap post globals := by
  intro following anchors stack
  simp only [List.append_assoc]
  apply (first (rest ++ following) anchors stack).bind
  rintro final ⟨afterEnv, after, rfl, holds⟩
  exact second afterEnv after holds following anchors stack

theorem returns (first : Continues module spec env body heap pre globals)
    (second : ∀ afterEnv after, pre afterEnv after → HeapMemoCalls.BlockReturnsRel module spec afterEnv rest after output globals) :
    HeapMemoCalls.BlockReturnsRel module spec env (body ++ rest) heap output globals := by
  intro anchors stack
  apply (first rest anchors stack).bind
  rintro final ⟨afterEnv, after, rfl, holds⟩
  exact second afterEnv after holds anchors stack

theorem assign (run : HeapMemoCalls.EvaluatesRel module spec env expression heap valuePost globals)
    (holds : ∀ value after, valuePost value after → post (setLocal env name value) after) :
    Continues module spec env [.assign [.name name] expression] heap post globals := by
  intro following anchors stack
  apply RunsRel.next (by trivial) rfl
  apply (run anchors (.assign name following env :: stack)).bind
  rintro final ⟨value, after, rfl, valueHolds⟩
  exact RunsRel.next (by trivial) rfl (RunsRel.refl ⟨setLocal env name value, after, rfl, holds value after valueHolds⟩)

theorem unpack (targets : Calls.targetNames expressions = some names)
    (run : HeapMemoCalls.EvaluatesRel module spec env expression heap valuePost globals)
    (holds : ∀ value after, valuePost value after → ∃ items finalHeap,
      Mutable.sequence value after = some (.ok items, finalHeap) ∧ names.length = items.length ∧
      Cache.Preserves spec after finalHeap ∧ post (assignNames env names items) finalHeap) :
    Continues module spec env [.assign [.tuple expressions] expression] heap post globals := by
  intro following anchors stack
  apply RunsRel.next (following := ⟨.expr expression env, .unpack names following env :: stack, heap⟩)
    (by trivial) (by simp [HeapCalls.step, targets])
  apply (run anchors _).bind
  rintro final ⟨value, after, rfl, valueHolds⟩
  obtain ⟨items, finalHeap, sequence, arity, preserved, result⟩ := holds value after valueHolds
  exact RunsRel.next (by trivial) (by simp [HeapCalls.step, HeapCalls.run, sequence, arity])
    (RunsRel.refl ⟨assignNames env names items, finalHeap, rfl, result⟩) preserved

end Continues

end ProjectEulerSolutions.HeapMemoBlock
