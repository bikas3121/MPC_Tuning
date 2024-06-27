% function to return NTF matrices:

function [Ar, Br, Cr, Dr] = ntf(Ah, Bh, Pf, Pg, Wf, Wg, L)

inv_Pg = inv(Pg);
Sf = Pf - inv_Pg;
inv_Sf = inv(Sf);
Ar = (Bh*Wf - inv_Pg*(L - Pg*Ah*Pf))*inv_Sf;
Br = Bh - inv_Pg*Wg;
Cr = Wf*inv_Sf;
Dr = 1;
end