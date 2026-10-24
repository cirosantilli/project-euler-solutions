import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9883_small : List Nat := []
theorem row9883_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9883 row9883_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9883_block_005 : List Nat := [9967, 9973]
def plan9883_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .possible, .possible, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9883_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9883 row9883_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9883_block_005
  · decide
  · rfl

def row9883_block_006 : List Nat := [10567, 10909, 10993, 11161, 12097]
def plan9883_block_006 : List Budget.Choice := [.first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 31), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .possible, .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 23), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9883_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9883 row9883_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9883_block_006
  · decide
  · rfl

def row9883_block_007 : List Nat := [12613, 13297]
def plan9883_block_007 : List Budget.Choice := [(.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), (.secondFactor 11), .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .possible, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .second, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .second, .first, .second, (.secondFactor 29), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 31), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row9883_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9883 row9883_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9883_block_007
  · decide
  · rfl

def row9883_block_008 : List Nat := [14557, 15121, 15259]
def plan9883_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .second, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .second, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9883_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9883 row9883_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9883_block_008
  · decide
  · rfl

def row9883_block_009 : List Nat := []
theorem row9883_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9883 row9883_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9883_block_010 : List Nat := []
theorem row9883_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9883 row9883_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9883_block_011 : List Nat := []
theorem row9883_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9883 row9883_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9883_block_012 : List Nat := []
theorem row9883_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9883 row9883_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9883_block_013 : List Nat := []
theorem row9883_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9883 row9883_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9883_large_left_left : List Nat := row9883_small ++ row9883_block_005
theorem row9883_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9883 row9883_large_left_left :=
  Budget.RowUpperBound.branch row9883_small_covers row9883_block_005_covers

def row9883_large_left_right_right : List Nat := row9883_block_007 ++ row9883_block_008
theorem row9883_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9883 row9883_large_left_right_right :=
  Budget.RowUpperBound.branch row9883_block_007_covers row9883_block_008_covers

def row9883_large_left_right : List Nat := row9883_block_006 ++ row9883_large_left_right_right
theorem row9883_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9883 row9883_large_left_right :=
  Budget.RowUpperBound.branch row9883_block_006_covers row9883_large_left_right_right_covers

def row9883_large_left : List Nat := row9883_large_left_left ++ row9883_large_left_right
theorem row9883_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9883 row9883_large_left :=
  Budget.RowUpperBound.branch row9883_large_left_left_covers row9883_large_left_right_covers

def row9883_large_right_left : List Nat := row9883_block_009 ++ row9883_block_010
theorem row9883_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9883 row9883_large_right_left :=
  Budget.RowUpperBound.branch row9883_block_009_covers row9883_block_010_covers

def row9883_large_right_right_right : List Nat := row9883_block_012 ++ row9883_block_013
theorem row9883_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9883 row9883_large_right_right_right :=
  Budget.RowUpperBound.branch row9883_block_012_covers row9883_block_013_covers

def row9883_large_right_right : List Nat := row9883_block_011 ++ row9883_large_right_right_right
theorem row9883_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9883 row9883_large_right_right :=
  Budget.RowUpperBound.branch row9883_block_011_covers row9883_large_right_right_right_covers

def row9883_large_right : List Nat := row9883_large_right_left ++ row9883_large_right_right
theorem row9883_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9883 row9883_large_right :=
  Budget.RowUpperBound.branch row9883_large_right_left_covers row9883_large_right_right_covers

def row9883_large : List Nat := row9883_large_left ++ row9883_large_right
theorem row9883_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9883 row9883_large :=
  Budget.RowUpperBound.branch row9883_large_left_covers row9883_large_right_covers

def row9883 : List Nat := row9883_large
theorem row9883_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9883 row9883 := row9883_large_covers

def row9887_small : List Nat := []
theorem row9887_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9887 row9887_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9887_block_005 : List Nat := [9923, 10301]
def plan9887_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .possible, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9887_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9887 row9887_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9887_block_005
  · decide
  · rfl

def row9887_block_006 : List Nat := [10589, 11261, 11909, 12107]
def plan9887_block_006 : List Budget.Choice := [(.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 29), (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 37), .second, (.secondFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9887_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9887 row9887_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9887_block_006
  · decide
  · rfl

def row9887_block_007 : List Nat := []
def plan9887_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 23), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 31), .first, (.firstFactor 31), .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .second, .first, .second, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9887_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9887 row9887_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9887_block_007
  · decide
  · rfl

def row9887_block_008 : List Nat := [15227]
def plan9887_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), (.secondFactor 13), .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .second, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .second, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), (.firstFactor 31), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9887_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9887 row9887_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9887_block_008
  · decide
  · rfl

def row9887_block_009 : List Nat := []
theorem row9887_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9887 row9887_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9887_block_010 : List Nat := []
theorem row9887_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9887 row9887_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9887_block_011 : List Nat := []
theorem row9887_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9887 row9887_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9887_block_012 : List Nat := []
theorem row9887_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9887 row9887_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9887_block_013 : List Nat := []
theorem row9887_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9887 row9887_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9887_large_left_left : List Nat := row9887_small ++ row9887_block_005
theorem row9887_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9887 row9887_large_left_left :=
  Budget.RowUpperBound.branch row9887_small_covers row9887_block_005_covers

def row9887_large_left_right_right : List Nat := row9887_block_007 ++ row9887_block_008
theorem row9887_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9887 row9887_large_left_right_right :=
  Budget.RowUpperBound.branch row9887_block_007_covers row9887_block_008_covers

def row9887_large_left_right : List Nat := row9887_block_006 ++ row9887_large_left_right_right
theorem row9887_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9887 row9887_large_left_right :=
  Budget.RowUpperBound.branch row9887_block_006_covers row9887_large_left_right_right_covers

def row9887_large_left : List Nat := row9887_large_left_left ++ row9887_large_left_right
theorem row9887_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9887 row9887_large_left :=
  Budget.RowUpperBound.branch row9887_large_left_left_covers row9887_large_left_right_covers

def row9887_large_right_left : List Nat := row9887_block_009 ++ row9887_block_010
theorem row9887_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9887 row9887_large_right_left :=
  Budget.RowUpperBound.branch row9887_block_009_covers row9887_block_010_covers

def row9887_large_right_right_right : List Nat := row9887_block_012 ++ row9887_block_013
theorem row9887_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9887 row9887_large_right_right_right :=
  Budget.RowUpperBound.branch row9887_block_012_covers row9887_block_013_covers

def row9887_large_right_right : List Nat := row9887_block_011 ++ row9887_large_right_right_right
theorem row9887_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9887 row9887_large_right_right :=
  Budget.RowUpperBound.branch row9887_block_011_covers row9887_large_right_right_right_covers

def row9887_large_right : List Nat := row9887_large_right_left ++ row9887_large_right_right
theorem row9887_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9887 row9887_large_right :=
  Budget.RowUpperBound.branch row9887_large_right_left_covers row9887_large_right_right_covers

def row9887_large : List Nat := row9887_large_left ++ row9887_large_right
theorem row9887_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9887 row9887_large :=
  Budget.RowUpperBound.branch row9887_large_left_covers row9887_large_right_covers

def row9887 : List Nat := row9887_large
theorem row9887_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9887 row9887 := row9887_large_covers

def row9901_small : List Nat := []
theorem row9901_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9901 row9901_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9901_block_005 : List Nat := [9931, 10321]
def plan9901_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 31), .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row9901_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9901 row9901_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9901_block_005
  · decide
  · rfl

def row9901_block_006 : List Nat := [10453, 11119, 11833, 11839, 11959]
def plan9901_block_006 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 29), (.firstFactor 29), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 29), .first, .first, .first, (.firstFactor 37), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .possible, .possible, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .possible, .first, (.firstFactor 37), .first, .first, (.firstFactor 31), .first, .second, .first, .first, (.secondFactor 29), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9901_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9901 row9901_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9901_block_006
  · decide
  · rfl

def row9901_block_007 : List Nat := [13219, 13759, 14401]
def plan9901_block_007 : List Budget.Choice := [(.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, (.firstFactor 19), .first, (.firstFactor 29), (.secondFactor 11), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .second, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .possible, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9901_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9901 row9901_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9901_block_007
  · decide
  · rfl

def row9901_block_008 : List Nat := [14887, 15901]
def plan9901_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 11), .second, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .second, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9901_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9901 row9901_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9901_block_008
  · decide
  · rfl

def row9901_block_009 : List Nat := []
theorem row9901_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9901 row9901_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9901_block_010 : List Nat := []
theorem row9901_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9901 row9901_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9901_block_011 : List Nat := []
theorem row9901_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9901 row9901_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9901_block_012 : List Nat := []
theorem row9901_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9901 row9901_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9901_block_013 : List Nat := []
theorem row9901_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9901 row9901_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9901_large_left_left : List Nat := row9901_small ++ row9901_block_005
theorem row9901_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9901 row9901_large_left_left :=
  Budget.RowUpperBound.branch row9901_small_covers row9901_block_005_covers

def row9901_large_left_right_right : List Nat := row9901_block_007 ++ row9901_block_008
theorem row9901_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9901 row9901_large_left_right_right :=
  Budget.RowUpperBound.branch row9901_block_007_covers row9901_block_008_covers

def row9901_large_left_right : List Nat := row9901_block_006 ++ row9901_large_left_right_right
theorem row9901_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9901 row9901_large_left_right :=
  Budget.RowUpperBound.branch row9901_block_006_covers row9901_large_left_right_right_covers

def row9901_large_left : List Nat := row9901_large_left_left ++ row9901_large_left_right
theorem row9901_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9901 row9901_large_left :=
  Budget.RowUpperBound.branch row9901_large_left_left_covers row9901_large_left_right_covers

def row9901_large_right_left : List Nat := row9901_block_009 ++ row9901_block_010
theorem row9901_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9901 row9901_large_right_left :=
  Budget.RowUpperBound.branch row9901_block_009_covers row9901_block_010_covers

def row9901_large_right_right_right : List Nat := row9901_block_012 ++ row9901_block_013
theorem row9901_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9901 row9901_large_right_right_right :=
  Budget.RowUpperBound.branch row9901_block_012_covers row9901_block_013_covers

def row9901_large_right_right : List Nat := row9901_block_011 ++ row9901_large_right_right_right
theorem row9901_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9901 row9901_large_right_right :=
  Budget.RowUpperBound.branch row9901_block_011_covers row9901_large_right_right_right_covers

def row9901_large_right : List Nat := row9901_large_right_left ++ row9901_large_right_right
theorem row9901_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9901 row9901_large_right :=
  Budget.RowUpperBound.branch row9901_large_right_left_covers row9901_large_right_right_covers

def row9901_large : List Nat := row9901_large_left ++ row9901_large_right
theorem row9901_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9901 row9901_large :=
  Budget.RowUpperBound.branch row9901_large_left_covers row9901_large_right_covers

def row9901 : List Nat := row9901_large
theorem row9901_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9901 row9901 := row9901_large_covers

def row9907_small : List Nat := []
theorem row9907_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9907 row9907_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9907_block_005 : List Nat := [9973]
def plan9907_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9907_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9907 row9907_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9907_block_005
  · decide
  · rfl

def row9907_block_006 : List Nat := [10771, 11863]
def plan9907_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 29), .second, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .possible, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9907_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9907 row9907_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9907_block_006
  · decide
  · rfl

def row9907_block_007 : List Nat := [12739, 13687]
def plan9907_block_007 : List Budget.Choice := [(.firstFactor 19), (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .second, .possible, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9907_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9907 row9907_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9907_block_007
  · decide
  · rfl

def row9907_block_008 : List Nat := [15259, 15493, 15619, 15937]
def plan9907_block_008 : List Budget.Choice := [(.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), (.secondFactor 11), .first, .first, .second, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 17), .first, .second, .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9907_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9907 row9907_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9907_block_008
  · decide
  · rfl

def row9907_block_009 : List Nat := []
theorem row9907_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9907 row9907_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9907_block_010 : List Nat := []
theorem row9907_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9907 row9907_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9907_block_011 : List Nat := []
theorem row9907_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9907 row9907_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9907_block_012 : List Nat := []
theorem row9907_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9907 row9907_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9907_block_013 : List Nat := []
theorem row9907_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9907 row9907_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9907_large_left_left : List Nat := row9907_small ++ row9907_block_005
theorem row9907_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9907 row9907_large_left_left :=
  Budget.RowUpperBound.branch row9907_small_covers row9907_block_005_covers

def row9907_large_left_right_right : List Nat := row9907_block_007 ++ row9907_block_008
theorem row9907_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9907 row9907_large_left_right_right :=
  Budget.RowUpperBound.branch row9907_block_007_covers row9907_block_008_covers

def row9907_large_left_right : List Nat := row9907_block_006 ++ row9907_large_left_right_right
theorem row9907_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9907 row9907_large_left_right :=
  Budget.RowUpperBound.branch row9907_block_006_covers row9907_large_left_right_right_covers

def row9907_large_left : List Nat := row9907_large_left_left ++ row9907_large_left_right
theorem row9907_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9907 row9907_large_left :=
  Budget.RowUpperBound.branch row9907_large_left_left_covers row9907_large_left_right_covers

def row9907_large_right_left : List Nat := row9907_block_009 ++ row9907_block_010
theorem row9907_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9907 row9907_large_right_left :=
  Budget.RowUpperBound.branch row9907_block_009_covers row9907_block_010_covers

def row9907_large_right_right_right : List Nat := row9907_block_012 ++ row9907_block_013
theorem row9907_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9907 row9907_large_right_right_right :=
  Budget.RowUpperBound.branch row9907_block_012_covers row9907_block_013_covers

def row9907_large_right_right : List Nat := row9907_block_011 ++ row9907_large_right_right_right
theorem row9907_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9907 row9907_large_right_right :=
  Budget.RowUpperBound.branch row9907_block_011_covers row9907_large_right_right_right_covers

def row9907_large_right : List Nat := row9907_large_right_left ++ row9907_large_right_right
theorem row9907_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9907 row9907_large_right :=
  Budget.RowUpperBound.branch row9907_large_right_left_covers row9907_large_right_right_covers

def row9907_large : List Nat := row9907_large_left ++ row9907_large_right
theorem row9907_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9907 row9907_large :=
  Budget.RowUpperBound.branch row9907_large_left_covers row9907_large_right_covers

def row9907 : List Nat := row9907_large
theorem row9907_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9907 row9907 := row9907_large_covers

end ProjectEulerSolutions.PrimePairLarge
