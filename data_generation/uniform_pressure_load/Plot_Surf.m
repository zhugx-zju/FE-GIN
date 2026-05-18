function Plot_Surf(contourShow,ux)
    surf(contourShow.PlotX,contourShow.PlotY,ux); view(0,90);
    shading flat;
    shading interp;
    colormap('jet');
    colorbar;
    set(gca,'YDir','normal');
    axis off;
    pause(0.1);
end