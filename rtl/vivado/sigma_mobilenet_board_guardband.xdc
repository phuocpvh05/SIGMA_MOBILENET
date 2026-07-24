# SIGMA MobileNet board-release timing guardband.
#
# The PS generates a realizable 250 MHz clock (4.000 ns).  This
# setup uncertainty reserves 0.200 ns inside that period, so a timing PASS
# requires the routed data paths to have at least 200 ps of physical margin.
# MOB7 also reserves 0.050 ns for hold.  The lower board clock and explicit
# setup/hold uncertainty keep the board-proven datapath away from its former
# 300 MHz edge.  XDC files accept constraint commands, not general Tcl
# control flow, so keep these as direct commands.
# PROCESSING_ORDER=LATE is assigned by the project/update scripts, after the PS
# IP creates clk_pl_0.
set_clock_uncertainty -setup 0.200 [get_clocks clk_pl_0]
set_clock_uncertainty -hold 0.050 [get_clocks clk_pl_0]
