import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows072

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7607_small : List Nat := PrimePairUpper.smallRow7607
theorem row7607_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7607 row7607_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7607_covers

def row7607_block_005 : List Nat := [8741]
def plan7607_block_005 : List Budget.Choice := [.first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .second, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row7607_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7607 row7607_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7607_block_005
  · decide
  · rfl

def row7607_block_006 : List Nat := [10739, 11411]
def plan7607_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .possible, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .possible, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 17), (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row7607_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7607 row7607_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7607_block_006
  · decide
  · rfl

def row7607_block_007 : List Nat := [13907]
def plan7607_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 31), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, .second, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .possible, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .second, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7607_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7607 row7607_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7607_block_007
  · decide
  · rfl

def row7607_block_008 : List Nat := [15161, 15377, 16067, 16127]
def plan7607_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 23), .first, (.firstFactor 7), .second, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 31), .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .possible, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row7607_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7607 row7607_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7607_block_008
  · decide
  · rfl

def row7607_block_009 : List Nat := [17093, 18041, 18191]
def plan7607_block_009 : List Budget.Choice := [.first, .second, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .second, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .second, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 19), .second, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7607_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7607 row7607_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7607_block_009
  · decide
  · rfl

def row7607_block_010 : List Nat := []
theorem row7607_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7607 row7607_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7607_block_011 : List Nat := []
theorem row7607_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7607 row7607_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7607_block_012 : List Nat := []
theorem row7607_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7607 row7607_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7607_block_013 : List Nat := []
theorem row7607_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7607 row7607_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7607_large_left_left : List Nat := row7607_small ++ row7607_block_005
theorem row7607_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7607 row7607_large_left_left :=
  Budget.RowUpperBound.branch row7607_small_covers row7607_block_005_covers

def row7607_large_left_right_right : List Nat := row7607_block_007 ++ row7607_block_008
theorem row7607_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7607 row7607_large_left_right_right :=
  Budget.RowUpperBound.branch row7607_block_007_covers row7607_block_008_covers

def row7607_large_left_right : List Nat := row7607_block_006 ++ row7607_large_left_right_right
theorem row7607_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7607 row7607_large_left_right :=
  Budget.RowUpperBound.branch row7607_block_006_covers row7607_large_left_right_right_covers

def row7607_large_left : List Nat := row7607_large_left_left ++ row7607_large_left_right
theorem row7607_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7607 row7607_large_left :=
  Budget.RowUpperBound.branch row7607_large_left_left_covers row7607_large_left_right_covers

def row7607_large_right_left : List Nat := row7607_block_009 ++ row7607_block_010
theorem row7607_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7607 row7607_large_right_left :=
  Budget.RowUpperBound.branch row7607_block_009_covers row7607_block_010_covers

def row7607_large_right_right_right : List Nat := row7607_block_012 ++ row7607_block_013
theorem row7607_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7607 row7607_large_right_right_right :=
  Budget.RowUpperBound.branch row7607_block_012_covers row7607_block_013_covers

def row7607_large_right_right : List Nat := row7607_block_011 ++ row7607_large_right_right_right
theorem row7607_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7607 row7607_large_right_right :=
  Budget.RowUpperBound.branch row7607_block_011_covers row7607_large_right_right_right_covers

def row7607_large_right : List Nat := row7607_large_right_left ++ row7607_large_right_right
theorem row7607_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7607 row7607_large_right :=
  Budget.RowUpperBound.branch row7607_large_right_left_covers row7607_large_right_right_covers

def row7607_large : List Nat := row7607_large_left ++ row7607_large_right
theorem row7607_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7607 row7607_large :=
  Budget.RowUpperBound.branch row7607_large_left_covers row7607_large_right_covers

def row7607 : List Nat := row7607_large
theorem row7607_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7607 row7607 := row7607_large_covers

def row7621_small : List Nat := PrimePairUpper.smallRow7621
theorem row7621_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7621 row7621_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7621_covers

def row7621_block_005 : List Nat := [9277, 9547, 9739]
def plan7621_block_005 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 37), .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .possible, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row7621_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7621 row7621_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7621_block_005
  · decide
  · rfl

def row7621_block_006 : List Nat := [11149, 12409]
def plan7621_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 19), .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 11), .first, .second, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 31), .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .second, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row7621_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7621 row7621_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7621_block_006
  · decide
  · rfl

def row7621_block_007 : List Nat := [12511, 12517, 12829, 13537, 13903, 14281, 14449]
def plan7621_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .possible, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 23), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .second, .possible, (.secondFactor 31), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row7621_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7621 row7621_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7621_block_007
  · decide
  · rfl

def row7621_block_008 : List Nat := [14713, 15091, 15889]
def plan7621_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 19), (.secondFactor 19), .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7621_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7621 row7621_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7621_block_008
  · decide
  · rfl

def row7621_block_009 : List Nat := [16633]
def plan7621_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 17), (.secondFactor 31), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7621_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7621 row7621_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7621_block_009
  · decide
  · rfl

def row7621_block_010 : List Nat := []
theorem row7621_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7621 row7621_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7621_block_011 : List Nat := []
theorem row7621_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7621 row7621_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7621_block_012 : List Nat := []
theorem row7621_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7621 row7621_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7621_block_013 : List Nat := []
theorem row7621_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7621 row7621_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7621_large_left_left : List Nat := row7621_small ++ row7621_block_005
theorem row7621_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7621 row7621_large_left_left :=
  Budget.RowUpperBound.branch row7621_small_covers row7621_block_005_covers

def row7621_large_left_right_right : List Nat := row7621_block_007 ++ row7621_block_008
theorem row7621_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7621 row7621_large_left_right_right :=
  Budget.RowUpperBound.branch row7621_block_007_covers row7621_block_008_covers

def row7621_large_left_right : List Nat := row7621_block_006 ++ row7621_large_left_right_right
theorem row7621_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7621 row7621_large_left_right :=
  Budget.RowUpperBound.branch row7621_block_006_covers row7621_large_left_right_right_covers

def row7621_large_left : List Nat := row7621_large_left_left ++ row7621_large_left_right
theorem row7621_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7621 row7621_large_left :=
  Budget.RowUpperBound.branch row7621_large_left_left_covers row7621_large_left_right_covers

def row7621_large_right_left : List Nat := row7621_block_009 ++ row7621_block_010
theorem row7621_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7621 row7621_large_right_left :=
  Budget.RowUpperBound.branch row7621_block_009_covers row7621_block_010_covers

def row7621_large_right_right_right : List Nat := row7621_block_012 ++ row7621_block_013
theorem row7621_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7621 row7621_large_right_right_right :=
  Budget.RowUpperBound.branch row7621_block_012_covers row7621_block_013_covers

def row7621_large_right_right : List Nat := row7621_block_011 ++ row7621_large_right_right_right
theorem row7621_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7621 row7621_large_right_right :=
  Budget.RowUpperBound.branch row7621_block_011_covers row7621_large_right_right_right_covers

def row7621_large_right : List Nat := row7621_large_right_left ++ row7621_large_right_right
theorem row7621_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7621 row7621_large_right :=
  Budget.RowUpperBound.branch row7621_large_right_left_covers row7621_large_right_right_covers

def row7621_large : List Nat := row7621_large_left ++ row7621_large_right
theorem row7621_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7621 row7621_large :=
  Budget.RowUpperBound.branch row7621_large_left_covers row7621_large_right_covers

def row7621 : List Nat := row7621_large
theorem row7621_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7621 row7621 := row7621_large_covers

def row7639_small : List Nat := PrimePairUpper.smallRow7639
theorem row7639_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7639 row7639_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7639_covers

def row7639_block_005 : List Nat := [8419, 8689, 9007, 9619, 9907]
def plan7639_block_005 : List Budget.Choice := [(.secondFactor 7), .possible, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 23), .second, (.firstFactor 17), .second, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 13), .possible, .first, .first, (.firstFactor 7), .second, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row7639_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7639 row7639_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7639_block_005
  · decide
  · rfl

def row7639_block_006 : List Nat := [10723, 11503, 11923]
def plan7639_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 17), (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .possible, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), .second, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row7639_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7639 row7639_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7639_block_006
  · decide
  · rfl

def row7639_block_007 : List Nat := [12511, 12739, 12979]
def plan7639_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 17), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 37), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .second]
set_option maxRecDepth 2048 in
theorem row7639_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7639 row7639_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7639_block_007
  · decide
  · rfl

def row7639_block_008 : List Nat := [15091, 15913]
def plan7639_block_008 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .second, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .possible, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 19), (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7639_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7639 row7639_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7639_block_008
  · decide
  · rfl

def row7639_block_009 : List Nat := [17137, 17359, 17383, 17599, 18289]
def plan7639_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 23), .possible, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7639_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7639 row7639_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7639_block_009
  · decide
  · rfl

def row7639_block_010 : List Nat := []
theorem row7639_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7639 row7639_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7639_block_011 : List Nat := []
theorem row7639_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7639 row7639_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7639_block_012 : List Nat := []
theorem row7639_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7639 row7639_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7639_block_013 : List Nat := []
theorem row7639_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7639 row7639_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7639_large_left_left : List Nat := row7639_small ++ row7639_block_005
theorem row7639_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7639 row7639_large_left_left :=
  Budget.RowUpperBound.branch row7639_small_covers row7639_block_005_covers

def row7639_large_left_right_right : List Nat := row7639_block_007 ++ row7639_block_008
theorem row7639_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7639 row7639_large_left_right_right :=
  Budget.RowUpperBound.branch row7639_block_007_covers row7639_block_008_covers

def row7639_large_left_right : List Nat := row7639_block_006 ++ row7639_large_left_right_right
theorem row7639_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7639 row7639_large_left_right :=
  Budget.RowUpperBound.branch row7639_block_006_covers row7639_large_left_right_right_covers

def row7639_large_left : List Nat := row7639_large_left_left ++ row7639_large_left_right
theorem row7639_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7639 row7639_large_left :=
  Budget.RowUpperBound.branch row7639_large_left_left_covers row7639_large_left_right_covers

def row7639_large_right_left : List Nat := row7639_block_009 ++ row7639_block_010
theorem row7639_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7639 row7639_large_right_left :=
  Budget.RowUpperBound.branch row7639_block_009_covers row7639_block_010_covers

def row7639_large_right_right_right : List Nat := row7639_block_012 ++ row7639_block_013
theorem row7639_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7639 row7639_large_right_right_right :=
  Budget.RowUpperBound.branch row7639_block_012_covers row7639_block_013_covers

def row7639_large_right_right : List Nat := row7639_block_011 ++ row7639_large_right_right_right
theorem row7639_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7639 row7639_large_right_right :=
  Budget.RowUpperBound.branch row7639_block_011_covers row7639_large_right_right_right_covers

def row7639_large_right : List Nat := row7639_large_right_left ++ row7639_large_right_right
theorem row7639_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7639 row7639_large_right :=
  Budget.RowUpperBound.branch row7639_large_right_left_covers row7639_large_right_right_covers

def row7639_large : List Nat := row7639_large_left ++ row7639_large_right
theorem row7639_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7639 row7639_large :=
  Budget.RowUpperBound.branch row7639_large_left_covers row7639_large_right_covers

def row7639 : List Nat := row7639_large
theorem row7639_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7639 row7639 := row7639_large_covers

def row7643_small : List Nat := PrimePairUpper.smallRow7643
theorem row7643_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7643 row7643_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7643_covers

def row7643_block_005 : List Nat := [8573, 8627, 8837, 9227, 9677, 10343]
def plan7643_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .possible, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .second, .first, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 13), (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .possible, (.secondFactor 13), .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row7643_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7643 row7643_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7643_block_005
  · decide
  · rfl

def row7643_block_006 : List Nat := [11987, 12239, 12323]
def plan7643_block_006 : List Budget.Choice := [(.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .second, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .possible, .first, (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7643_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7643 row7643_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7643_block_006
  · decide
  · rfl

def row7643_block_007 : List Nat := [12743, 13331, 13679, 13721]
def plan7643_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 11), .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .second, .first, .possible, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7643_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7643 row7643_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7643_block_007
  · decide
  · rfl

def row7643_block_008 : List Nat := [14783, 15077, 15473]
def plan7643_block_008 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .second, (.firstFactor 31), .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row7643_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7643 row7643_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7643_block_008
  · decide
  · rfl

def row7643_block_009 : List Nat := [16691]
def plan7643_block_009 : List Budget.Choice := [.first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7643_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7643 row7643_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7643_block_009
  · decide
  · rfl

def row7643_block_010 : List Nat := []
theorem row7643_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7643 row7643_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7643_block_011 : List Nat := []
theorem row7643_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7643 row7643_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7643_block_012 : List Nat := []
theorem row7643_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7643 row7643_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7643_block_013 : List Nat := []
theorem row7643_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7643 row7643_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7643_large_left_left : List Nat := row7643_small ++ row7643_block_005
theorem row7643_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7643 row7643_large_left_left :=
  Budget.RowUpperBound.branch row7643_small_covers row7643_block_005_covers

def row7643_large_left_right_right : List Nat := row7643_block_007 ++ row7643_block_008
theorem row7643_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7643 row7643_large_left_right_right :=
  Budget.RowUpperBound.branch row7643_block_007_covers row7643_block_008_covers

def row7643_large_left_right : List Nat := row7643_block_006 ++ row7643_large_left_right_right
theorem row7643_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7643 row7643_large_left_right :=
  Budget.RowUpperBound.branch row7643_block_006_covers row7643_large_left_right_right_covers

def row7643_large_left : List Nat := row7643_large_left_left ++ row7643_large_left_right
theorem row7643_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7643 row7643_large_left :=
  Budget.RowUpperBound.branch row7643_large_left_left_covers row7643_large_left_right_covers

def row7643_large_right_left : List Nat := row7643_block_009 ++ row7643_block_010
theorem row7643_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7643 row7643_large_right_left :=
  Budget.RowUpperBound.branch row7643_block_009_covers row7643_block_010_covers

def row7643_large_right_right_right : List Nat := row7643_block_012 ++ row7643_block_013
theorem row7643_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7643 row7643_large_right_right_right :=
  Budget.RowUpperBound.branch row7643_block_012_covers row7643_block_013_covers

def row7643_large_right_right : List Nat := row7643_block_011 ++ row7643_large_right_right_right
theorem row7643_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7643 row7643_large_right_right :=
  Budget.RowUpperBound.branch row7643_block_011_covers row7643_large_right_right_right_covers

def row7643_large_right : List Nat := row7643_large_right_left ++ row7643_large_right_right
theorem row7643_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7643 row7643_large_right :=
  Budget.RowUpperBound.branch row7643_large_right_left_covers row7643_large_right_right_covers

def row7643_large : List Nat := row7643_large_left ++ row7643_large_right
theorem row7643_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7643 row7643_large :=
  Budget.RowUpperBound.branch row7643_large_left_covers row7643_large_right_covers

def row7643 : List Nat := row7643_large
theorem row7643_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7643 row7643 := row7643_large_covers

end ProjectEulerSolutions.PrimePairLarge
