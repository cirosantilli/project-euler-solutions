import solvers.«79.iterator.entry.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.iterator.entry.source»
import lang.py.FormalSW.RuntimeExceptionObservation

/-! Complete calls agree on input failures, including acquisition effects and caller-visible object graphs. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem iterator_rejection_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String) (s : String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (noGrowth : yielded.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (invalid : ¬ ValidAttempt s) :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftLayout, leftRun⟩ := solve_iterator_rejection context solverExtension source choices
    receiver iterator base acquired middle yielded acquiredUsed nextUsed finalUsed closed acquisition attempts s
      trace ran noGrowth valid invalid
  have rejected : s.length ≠ 3 ∨ Strings.isDigit s = false := by
    by_cases length : s.length = 3
    · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
    · exact Or.inl length
  obtain ⟨right, rightPrivate, rightRun⟩ := p79.naive_iterator_rejection context referenceExtension source choices
    receiver iterator base acquired middle yielded acquiredUsed nextUsed finalUsed closed acquisition attempts s
      trace ran noGrowth valid rejected
  exact Mutable.Call.equiv_of_finishes leftRun rightRun
    (leftLayout.equivalent_exceptions rightPrivate.layout base.length (by omega)
      (fun _ _ => rfl) (fun _ _ => rfl) (invalidException s))

theorem iterator_failure_equiv (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired yielded : Mutable.Heap)
    (acquiredUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String) (error : ExceptionObject)
    (trace : Traversal context choices iterator graph.text acquired acquiredUsed attempts (.error error) yielded finalUsed)
    (noGrowth : yielded.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ((IteratorCalls.call context "solve" [receiver] base).run choices).Equiv
      ((IteratorCalls.call context "naive" [receiver] base).run choices) := by
  obtain ⟨left, leftLayout, leftRun⟩ := solve_iterator_failure context solverExtension source choices
    receiver iterator base acquired yielded acquiredUsed finalUsed closed acquisition attempts error trace noGrowth valid
  obtain ⟨right, rightPrivate, rightRun⟩ := p79.naive_iterator_failure context referenceExtension source choices
    receiver iterator base acquired yielded acquiredUsed finalUsed closed acquisition attempts error trace noGrowth valid
  exact Mutable.Call.equiv_of_finishes leftRun rightRun
    (leftLayout.equivalent_exceptions rightPrivate.layout base.length (by omega)
      (fun _ _ => rfl) (fun _ _ => rfl) error)

end solvers.«79».source
